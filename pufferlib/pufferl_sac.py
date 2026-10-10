"""Feed-forward discrete SAC using the same evaluate/train lifecycle as PuffeRL.

Drive networks, replay placement, pre-reset observations, BF16 autocast, compile,
DDP, Hydra startup, benchmark evaluation, replay rendering and experiment commands
are integrated. Actor export remains pending; recurrence and exact replay/environment
resume remain deferred.
"""

from collections import defaultdict
import copy
from contextlib import nullcontext
import glob
import math
import numbers
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
DDP = torch.nn.parallel.DistributedDataParallel
DEFAULT_CONFIG = dict(
    env="puffer_drive", device="cpu", seed=0, run_name="sac", batch_size=10_000,
    minibatch_size=256, max_minibatch_size=256, update_epochs=1,
    total_timesteps=None, gamma=0.99, learning_starts=10_000, replay_update_ratio=0.1,
    learning_rate=3e-4, critic_learning_rate=3e-4, alpha_learning_rate=3e-4,
    optimizer="adam", adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, adam_weight_decay=0.0,
    max_grad_norm=None, amp=False, precision="float32", compile=False, compile_mode="default",
    compile_fullgraph=False, cpu_offload=True,
    initial_alpha=0.2, target_entropy=None, tau=0.005, torch_deterministic=False,
    anneal_lr=False, checkpoint_interval=50, final_model_name="final_model.pt", resume_state_path=None,
    fresh_run=False,
)

SAC_TRAIN_KEYS = frozenset(DEFAULT_CONFIG) - {"env"} | {"data_dir"}
SAC_EVALUATION_KEYS = frozenset({"evaluation_interval_epochs", "evaluation_benchmarks"})
DEFAULT_MINIBATCH_SIZE = 256
PPO_CHECKPOINT_ENV_KEYS = frozenset({
    "action_type",
    "dynamics_model",
    "goal_source",
    "goal_regen_mode",
    "num_goals",
    "min_goal_spacing",
    "max_goal_spacing",
    "obs_goal_lane_distance",
    "reward_conditioning",
    "reward_randomization",
    "trajectory_prediction_length",
    "num_trajectory_scaling_factors",
    "trajectory_scaling_factors",
    "obs_slots_boundary_n",
    "obs_slots_lane_n",
    "obs_boundary_stride",
    "obs_lane_stride",
    "obs_dropout_boundary",
    "obs_dropout_lane",
    "partner_blindness_duration_seconds",
    "partner_blindness_prob",
    "partner_blindness_trigger_prob",
    "phantom_braking_duration_seconds",
    "phantom_braking_prob",
    "phantom_braking_trigger_prob",
    "obs_slots_partners_n",
    "obs_slots_traffic_controls_n",
    "traffic_control_scope",
})


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
            if train_key not in SAC_TRAIN_KEYS | SAC_EVALUATION_KEYS:
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
    args["train"] = {
        key: value for key, value in inherited_train.items()
        if key in SAC_TRAIN_KEYS | SAC_EVALUATION_KEYS
    }
    args["env_name"] = env_name
    return args


def _resolve_resume_state_path(train_config, load_model_path=None, load_id=None):
    """Resolve an explicit learner resume or same-run local autodiscovery."""
    resume_state_path = train_config.get("resume_state_path")
    if resume_state_path is not None and (load_model_path is not None or load_id is not None):
        raise ValueError("Cannot combine train.resume_state_path with an actor warmstart")
    if load_model_path is not None and load_id is not None:
        raise ValueError("Choose one actor warmstart source: load_model_path or load_id")
    if load_model_path == "latest":
        raise ValueError(
            "SAC requires an explicit local load_model_path actor artifact; the 'latest' shorthand is unsupported."
        )
    if load_id is not None and (not isinstance(load_id, str) or not load_id.strip()):
        raise ValueError("SAC remote model loading requires a non-empty load_id")
    fresh_run = train_config.get("fresh_run", False)
    if not isinstance(fresh_run, bool):
        raise ValueError("train.fresh_run must be a boolean")
    auto_resume = False
    if resume_state_path is None and load_model_path is None and load_id is None and not fresh_run:
        data_dir = train_config.get("data_dir")
        if not data_dir:
            raise ValueError("SAC requires an explicit data_dir")
        discovered_path = Path(data_dir) / "trainer_state.pt"
        if discovered_path.is_file():
            resume_state_path = str(discovered_path)
            auto_resume = True
    return resume_state_path, auto_resume


def _download_remote_model(args, global_rank, world_size):
    """Download an actor once and share its path with the learner ranks."""
    if world_size == 1:
        from pufferlib.sac_logging import download_model

        return download_model(args)

    payload = [None]
    download_error = None
    if global_rank == 0:
        try:
            from pufferlib.sac_logging import download_model

            payload[0] = (True, download_model(args))
        except Exception as error:
            download_error = error
            payload[0] = (False, f"{type(error).__name__}: {error}")
    torch.distributed.broadcast_object_list(payload, src=0)
    succeeded, result = payload[0]
    if not succeeded:
        if download_error is not None:
            raise download_error
        raise RuntimeError(f"Remote SAC model download failed on rank 0: {result}")
    return result


def _restore_checkpoint_architecture(startup_args, source_path, ppo):
    """Restore source network settings and PPO's observation whitelist."""
    source_args_for_loading = copy.deepcopy(startup_args)
    source_args_for_loading["load_model_path"] = source_path
    source_args, _ = ppo.drive_benchmark.load_checkpoint_architecture(source_args_for_loading)
    if source_args["policy_name"] != "DriveSAC":
        raise ValueError(
            f"SAC checkpoint policy_name must be 'DriveSAC', got {source_args['policy_name']!r}"
        )
    if source_args["rnn_name"] is not None:
        raise NotImplementedError("SAC checkpoint recurrence is deferred; expected rnn_name=null")

    startup_args["policy"].update(source_args["policy"])
    startup_args["rnn"].update(source_args["rnn"])
    startup_args["policy_name"] = source_args["policy_name"]
    startup_args["rnn_name"] = source_args["rnn_name"]
    for key in PPO_CHECKPOINT_ENV_KEYS:
        if key in source_args["env"]:
            startup_args["env"][key] = source_args["env"][key]


def _load_actor_weights(policy, model_path, device, ppo):
    """Load a local actor artifact without restoring any SAC learner state."""
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    policy.load_state_dict(ppo.clean_policy_state_dict(state_dict))


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
    if args.get("load_model_path") is not None:
        _load_actor_weights(actor, args["load_model_path"], device, ppo)
    return actor


def eval(env_name="puffer_drive", args=None, benchmark_names=None):
    """Run Drive benchmarks with a SAC actor and optional value-free replay capture."""
    if env_name != "puffer_drive":
        raise NotImplementedError(f"Drive SAC evaluation is not configured for {env_name!r}")
    args = copy.deepcopy(load_config(env_name) if args is None else args)
    eval_config = args["eval"]
    rendering_requested = (
        eval_config["render_scenarios"]
        or eval_config["render_filter"] is not None
        or eval_config["failure_replay_csv"] is not None
        or eval_config["capture_observations"]
    )

    model_path = args.get("load_model_path")
    load_id = args.get("load_id")
    if model_path is not None and load_id is not None:
        import pufferlib
        raise pufferlib.APIUsageError("Choose one SAC eval model source: load_model_path or load_id")
    if model_path is None and load_id is None:
        import pufferlib
        raise pufferlib.APIUsageError("SAC eval requires load_model_path or load_id")

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        from pufferlib import pufferl as ppo
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
    if load_id is not None:
        model_path = _download_remote_model(args, global_rank=0, world_size=1)
        args["load_model_path"] = model_path
    if model_path == "latest":
        import pufferlib
        raise pufferlib.APIUsageError("SAC eval requires an explicit local load_model_path checkpoint")

    _restore_checkpoint_architecture(args, model_path, ppo)
    if args["policy"].get("action_type") != "discrete":
        raise ValueError("SAC eval requires policy.action_type=discrete")
    if args["policy"].get("shared_network") is not False:
        raise ValueError("SAC eval requires policy.shared_network=false")

    selected_benchmarks = benchmark_names if benchmark_names is not None else eval_config["benchmarks"]
    environment_config, benchmarks = ppo.drive_benchmark.load_benchmark_config(
        eval_config["benchmark_config"], selected_benchmarks,
    )
    actor_args = ppo.drive_benchmark.build_benchmark_args(args, benchmarks[0], environment_config)
    actor_args["env"].update(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
    )
    actor_args["vec"].update(
        backend="Serial",
        num_envs=1,
        num_workers=1,
        batch_size=1,
    )
    actor_vecenv = ppo.load_env(env_name, actor_args, seed=actor_args["vec"].get("seed"))
    try:
        policy = load_policy(actor_args, actor_vecenv, env_name)
    finally:
        actor_vecenv.close()

    evaluator = ppo.eval
    if rendering_requested:
        from pufferlib.sac_evaluation import eval as evaluator
    return evaluator(
        env_name=env_name,
        args=args,
        policy=policy,
        benchmark_names=benchmark_names,
    )


def _initialize_process_group(device, pufferl):
    launcher_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    requested_device = torch.device(pufferl.torch_device(device))
    owns_process_group = False
    resolved_device = device
    if not torch.distributed.is_initialized() and launcher_world_size > 1:
        if requested_device.type == "cuda":
            torch.cuda.set_device(local_rank)
            resolved_device = f"cuda:{local_rank}"
            backend = "nccl"
        else:
            backend = "gloo"
        torch.distributed.init_process_group(backend=backend)
        owns_process_group = True
    elif torch.distributed.is_initialized() and requested_device.type == "cuda":
        torch.cuda.set_device(local_rank)
        if torch.distributed.get_world_size() > 1:
            resolved_device = f"cuda:{local_rank}"

    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    return owns_process_group, resolved_device, world_size, rank


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
    every received row; global_step counts rank-local received slots and agent_steps
    is their synchronized global total. Both include reset, masked, and valid rows
    discarded after the insertion quota is reached.
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
        if config["precision"] not in ("float32", "bfloat16"):
            raise ValueError("Invalid precision: use float32 or bfloat16")
        if config["precision"] == "bfloat16" and not config["amp"]:
            raise ValueError("bfloat16 precision requires train.amp=True")
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
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.distributed = self.world_size > 1
        torch_seed, _ = pufferl.derive_rank_seeds(None, config["seed"], self.world_size, self.rank)
        reset_seed = config["seed"]
        if self.distributed:
            _, reset_seed = pufferl.derive_rank_seeds(
                config["seed"], config["seed"], self.world_size, self.rank,
            )
        random.seed(torch_seed)
        np.random.seed(torch_seed % 2**32)
        torch.manual_seed(torch_seed)

        self.helpers = pufferl
        self.pytorch = pytorch
        self.config = config
        self.device = torch.device(pufferl.torch_device(config["device"]))
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

        if config["precision"] == "bfloat16":
            if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
                raise pufferl.pufferlib.APIUsageError(
                    "bfloat16 precision requires a CUDA device with bf16 support"
                )
            if self.device.type not in ("cpu", "cuda"):
                raise pufferl.pufferlib.APIUsageError(
                    f"bfloat16 precision is unsupported on device type {self.device.type}"
                )
        self.amp_context = nullcontext()
        if config["amp"] and config["precision"] == "bfloat16":
            self.amp_context = torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16)

        self.total_agents = vecenv.num_agents
        self.minibatch_size = config["minibatch_size"]
        self.max_minibatch_size = config["max_minibatch_size"]
        self.observations = torch.zeros(
            self.total_agents, *vecenv.single_observation_space.shape, device=self.device, dtype=torch.float32
        )
        self.actions = torch.zeros(self.total_agents, device=self.device, dtype=torch.int64)
        self.pending = torch.zeros(self.total_agents, device=self.device, dtype=torch.bool)
        self.replay = replay

        self.uncompiled_policy = policy.to(self.device, dtype=torch.float32)
        self.model_size = sum(
            parameter.numel() for parameter in self.uncompiled_policy.parameters() if parameter.requires_grad
        )
        self.uncompiled_critic = critic.to(self.device, dtype=torch.float32)

        if self.distributed:
            ddp_kwargs = {}
            if self.device.type == "cuda":
                local_rank = torch.cuda.current_device()
                ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)
            self.ddp_policy = DDP(self.uncompiled_policy, **ddp_kwargs)
            self.ddp_critic = DDP(self.uncompiled_critic, **ddp_kwargs)
        else:
            self.ddp_policy = self.uncompiled_policy
            self.ddp_critic = self.uncompiled_critic
        self.policy_train = self.ddp_policy
        self.critic_train = self.ddp_critic

        # DDP synchronizes the online critic from rank 0 during construction.
        self.uncompiled_target_critic = copy.deepcopy(self.uncompiled_critic).requires_grad_(False).eval()
        self.policy = self.uncompiled_policy
        self.critic = self.uncompiled_critic
        self.target_critic = self.uncompiled_target_critic
        if config["compile"]:
            compile_kwargs = {
                "mode": config["compile_mode"],
                "fullgraph": config["compile_fullgraph"],
            }
            self.policy = torch.compile(self.uncompiled_policy, **compile_kwargs)
            self.critic = torch.compile(self.uncompiled_critic, **compile_kwargs)
            self.target_critic = torch.compile(self.uncompiled_target_critic, **compile_kwargs)
            if self.distributed:
                self.policy_train = torch.compile(self.ddp_policy, **compile_kwargs)
                self.critic_train = torch.compile(self.ddp_critic, **compile_kwargs)
            else:
                self.policy_train = self.policy
                self.critic_train = self.critic
        self.log_alpha = torch.nn.Parameter(torch.tensor(
            math.log(config["initial_alpha"]), device=self.device, dtype=torch.float32
        ))
        self.optimizer = self._make_model_optimizer(self.uncompiled_policy.parameters(), config["learning_rate"])
        self.critic_optimizer = self._make_model_optimizer(
            self.uncompiled_critic.parameters(), config["critic_learning_rate"]
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config["alpha_learning_rate"])
        scheduler_steps = max(1, math.ceil(
            config["total_timesteps"] / (self.world_size * self.refresh_count)
        ))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=scheduler_steps)
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=scheduler_steps
        )
        self.total_epochs = scheduler_steps
        if self.rank != 0:
            logger = None
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
            vecenv.async_reset(reset_seed)
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

    def load_training_state(self, path, same_run=False):
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
            "checkpoint_interval", "resume_state_path", "amp", "precision", "compile",
            "compile_mode", "compile_fullgraph", "cpu_offload", "fresh_run",
        }
        saved_config = {**DEFAULT_CONFIG, **state["config"]}
        if same_run and saved_config["run_name"] != self.config["run_name"]:
            raise ValueError(
                "Auto-discovered SAC checkpoint run_name does not match the target run_name"
            )
        for key, value in saved_config.items():
            if key not in operational_config and self.config[key] != value:
                raise ValueError(f"Checkpoint config mismatch: {key}")
        for name in ("policy", "critic", "target_critic"):
            model_state = self.helpers.clean_policy_state_dict(state[f"{name}_state_dict"])
            getattr(self, f"uncompiled_{name}").load_state_dict(model_state)
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        for name in ("optimizer", "critic_optimizer", "alpha_optimizer"):
            getattr(self, name).load_state_dict(state[f"{name}_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.critic_scheduler.load_state_dict(state["critic_scheduler_state_dict"])
        self.total_epochs = self.scheduler.T_max
        for name in ("epoch", "valid_inserts", "learner_updates", "sampled_transitions", "best_score"):
            setattr(self, name, state[name])
        self.agent_steps = state.get("agent_steps", state["global_step"])
        local_step, remainder = divmod(self.agent_steps, self.world_size)
        self.global_step = local_step + int(self.rank < remainder)
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
        if self.rank != 0:
            torch_seed, _ = self.helpers.derive_rank_seeds(
                None, self.config["seed"], self.world_size, self.rank,
            )
            resume_seed = int(np.random.SeedSequence([torch_seed, self.epoch, self.agent_steps]).generate_state(1)[0])
            random.seed(resume_seed)
            np.random.seed(resume_seed)
            torch.manual_seed(resume_seed)
        if self.rank == 0:
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
        if self.device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
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
        received_budget = max(
            0, math.ceil((self.config["total_timesteps"] - self.agent_steps) / self.world_size)
        )
        received_count_total = 0
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
            received_count_total += received_count
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
            with self.amp_context:
                logits = self.policy(observations)
            actions, _, _, continuous_actions = self.pytorch.sample_logits(
                logits.float(), env_continuous=self.env_continuous, policy=self.uncompiled_policy
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
            if valid_inserts >= insert_target or received_count_total >= received_budget:
                break

        if self.distributed:
            deltas = torch.tensor(
                [received_count_total, valid_inserts],
                device=self.device,
                dtype=torch.int64,
            )
            torch.distributed.all_reduce(deltas, op=torch.distributed.ReduceOp.SUM)
            received_count_total, valid_inserts = deltas.tolist()
        self.agent_steps += received_count_total
        self.valid_inserts += valid_inserts
        profile.end()
        return self.helpers.pufferlib.utils.reduce_environment_metrics(self.stats)

    def train(self):
        """Optimize from replay after evaluate(); one call increments epoch once."""
        if self.device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        profile("train_misc", epoch, nest=True)
        losses = defaultdict(float)
        replay_ready = True
        if self.distributed:
            ready = torch.tensor(
                int(len(self.replay) == self.batch_size),
                device=self.device,
                dtype=torch.int32,
            )
            torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
            replay_ready = bool(ready.item())
        updates = self._train_sac_transition(losses, profile, epoch) if replay_ready else 0
        profile("train_misc", epoch)
        if updates and self.config["anneal_lr"]:
            if self.scheduler.last_epoch < self.scheduler.T_max:
                self.scheduler.step()
            if self.critic_scheduler.last_epoch < self.critic_scheduler.T_max:
                self.critic_scheduler.step()
        profile.end()

        self.losses = losses
        self.epoch += 1
        done_training = self.agent_steps >= self.config["total_timesteps"]
        logs = None
        log_decision = self.rank == 0 and (
            done_training or self.agent_steps == 0 or time.time() > self.last_log_time + 0.25
        )
        if self.helpers.dist_sum(int(log_decision), self.device):
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
        with self.amp_context:
            logits = self.policy(mb_next_obs)
            next_q1, next_q2 = self.target_critic(mb_next_obs)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        probs = logprobs.exp()
        next_q1 = next_q1.float()
        next_q2 = next_q2.float()
        next_value = (probs * (torch.minimum(next_q1, next_q2) - self.alpha * logprobs)).sum(-1)
        return mb_rewards + mb_discounts * next_value

    def _critic_loss(self, mb_obs, mb_actions, mb_targets):
        with self.amp_context:
            q1, q2 = self.critic_train(mb_obs)
        q1 = q1.float()
        q2 = q2.float()
        q1 = q1.gather(1, mb_actions.unsqueeze(-1)).squeeze(-1)
        q2 = q2.gather(1, mb_actions.unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(q1, mb_targets) + F.mse_loss(q2, mb_targets)
        return loss, q1.detach().mean(), q2.detach().mean()

    def _policy_loss(self, mb_obs):
        with self.amp_context:
            logits = self.policy_train(mb_obs)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        probs = logprobs.exp()
        with torch.no_grad(), self.amp_context:
            q1, q2 = self.critic(mb_obs)
        q1 = q1.float()
        q2 = q2.float()
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
                    tensor.to(self.device, non_blocking=True)
                    for tensor in self.replay.get_batch(micro_indices)
                )
                profile("train_forward", epoch)
                last_microbatch = micro_start + micro_count == logical_batch_count
                sync_context = (
                    self.ddp_critic.no_sync()
                    if self.distributed and not last_microbatch
                    else nullcontext()
                )
                with sync_context:
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
                mb_obs = self.replay.get_batch(micro_indices)[0].to(self.device, non_blocking=True)
                profile("train_forward", epoch)
                last_microbatch = micro_start + micro_count == logical_batch_count
                sync_context = (
                    self.ddp_policy.no_sync()
                    if self.distributed and not last_microbatch
                    else nullcontext()
                )
                with sync_context:
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
            if self.distributed:
                torch.distributed.all_reduce(self.log_alpha.grad, op=torch.distributed.ReduceOp.SUM)
                self.log_alpha.grad.div_(self.world_size)
            self.alpha_optimizer.step()

            profile("train_misc", epoch)
            # Polyak-update all target parameters and buffers, including encoders.
            with torch.no_grad():
                online = dict(self.uncompiled_critic.named_parameters()) | dict(
                    self.uncompiled_critic.named_buffers()
                )
                targets = dict(self.uncompiled_target_critic.named_parameters()) | dict(
                    self.uncompiled_target_critic.named_buffers()
                )
                for key, target in targets.items():
                    if target.is_floating_point():
                        target.lerp_(online[key], self.config["tau"])
                    else:
                        target.copy_(online[key])

            self.learner_updates += 1
            self.sampled_transitions += logical_batch_count * self.world_size
            update_losses = dict(
                critic_loss=critic_loss_value, policy_loss=policy_loss_value,
                alpha_loss=alpha_loss.detach().item(), entropy=mean_entropy.item(), alpha=self.alpha.item(),
                q1=q1_value, q2=q2_value, target=target_value,
            )
            profile("train_misc", epoch)
            for key, value in update_losses.items():
                losses[key] += value * logical_batch_count / total_sampled_transitions
        return total_minibatches

    def _reduce_environment_metrics(self):
        if not self.distributed:
            return self.helpers.pufferlib.utils.reduce_environment_metrics(self.stats)
        local_totals = {}
        for key, values in self.stats.items():
            if values and isinstance(values[0], numbers.Number):
                local_totals[key] = (float(np.sum(values)), len(values))
        gathered_keys = [None] * self.world_size
        torch.distributed.all_gather_object(gathered_keys, list(local_totals))
        metric_keys = sorted({key for rank_keys in gathered_keys for key in rank_keys})
        if not metric_keys:
            return {}
        metric_totals = torch.zeros(len(metric_keys) * 2, device=self.device, dtype=torch.float64)
        for index, key in enumerate(metric_keys):
            if key in local_totals:
                metric_totals[index] = local_totals[key][0]
                metric_totals[len(metric_keys) + index] = local_totals[key][1]
        torch.distributed.all_reduce(metric_totals, op=torch.distributed.ReduceOp.SUM)
        reduced_metrics = {}
        total_distance = None
        total_infractions = None
        for index, key in enumerate(metric_keys):
            value_sum = metric_totals[index].item()
            value_count = metric_totals[len(metric_keys) + index].item()
            if key == "total_distance_travelled_sum":
                total_distance = value_sum
            elif key == "total_infraction_count":
                total_infractions = value_sum
            else:
                reduced_metrics[key] = value_sum / max(value_count, 1)
        if total_distance is not None and total_infractions is not None:
            reduced_metrics["total_distance_travelled"] = total_distance
            reduced_metrics["total_infractions"] = total_infractions
            reduced_metrics["avg_distance_per_infraction"] = total_distance / max(total_infractions, 1.0)
        return reduced_metrics

    def mean_and_log(self):
        self.stats = self._reduce_environment_metrics()
        self.last_stats = dict(self.stats)
        self.last_stats_epoch = self.epoch
        if self.distributed:
            self.losses = {
                key: self.helpers.dist_mean(value, self.device)
                for key, value in sorted(self.losses.items())
            }
        performance = {key: value["elapsed"] for key, value in self.profile}
        logs = {
            "SPS": self.helpers.dist_sum(self.sps, self.device),
            "agent_steps": self.agent_steps,
            "uptime": self.uptime,
            "epoch": self.epoch,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "replay/size": len(self.replay),
            "replay/valid_inserts": self.valid_inserts,
            "replay/sampled_transitions": self.sampled_transitions,
            "sac/learner_updates": self.learner_updates,
            **{f"environment/{key}": value for key, value in self.stats.items()},
            **{f"losses/{key}": value for key, value in self.losses.items()},
            **{f"performance/{key}": value for key, value in performance.items()},
        }
        if self.rank != 0:
            return None
        self.logger.log(logs, self.agent_steps)
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

        if self.rank != 0:
            return None
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
        if self.rank != 0:
            return None
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
            "critic_state_dict": self.uncompiled_critic.state_dict(),
            "target_critic_state_dict": self.uncompiled_target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
            "global_step": self.global_step,
            "agent_steps": self.agent_steps,
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
    all_logs = []
    cleanup_logger = logger
    close_started = False
    owns_process_group = False
    auto_resume = False
    experiment_config = None
    load_model_path = None
    training_evaluation_scheduled = False
    last_training_evaluation_epoch = None
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
            unsupported_train_keys = set(startup_args["train"]) - SAC_TRAIN_KEYS - SAC_EVALUATION_KEYS
            if unsupported_train_keys:
                unsupported_names = ", ".join(sorted(unsupported_train_keys))
                raise ValueError(
                    f"Unsupported SAC train settings: {unsupported_names}. "
                    "PPO-only train settings are not used by SAC."
                )
            if startup_args["train"].get("evaluation_interval_epochs") is not None:
                training_evaluation_scheduled = ppo.drive_benchmark.validate_training_evaluation_config(startup_args)
            train_args = startup_args["train"]
            load_model_path = startup_args.get("load_model_path")
            resume_state_path, auto_resume = _resolve_resume_state_path(
                train_args, load_model_path, startup_args.get("load_id"),
            )
            train_args["resume_state_path"] = resume_state_path
            owns_process_group, train_args["device"], world_size, global_rank = _initialize_process_group(
                train_args["device"], ppo,
            )
            if global_rank != 0:
                logger = None
                cleanup_logger = None
            if startup_args.get("load_id") is not None:
                load_model_path = _download_remote_model(startup_args, global_rank, world_size)
                startup_args["load_model_path"] = load_model_path
            architecture_source_path = resume_state_path or load_model_path
            if architecture_source_path is not None:
                _restore_checkpoint_architecture(startup_args, architecture_source_path, ppo)
            if startup_args.get("policy_name") != "DriveSAC":
                raise ValueError(
                    f"SAC requires policy_name='DriveSAC', got {startup_args.get('policy_name')!r}"
                )
            if startup_args.get("rnn_name") is not None:
                raise NotImplementedError("SAC recurrence is deferred; set rnn_name=null")
            if startup_args["policy"].get("action_type") != "discrete":
                raise ValueError("SAC requires policy.action_type=discrete")
            if startup_args["policy"].get("shared_network") is not False:
                raise ValueError("SAC requires policy.shared_network=false for independent actor and critic networks")

            run_name = startup_args.get("run_name", train_args.get("run_name", "sac"))
            train_seed = train_args.get("seed")
            if train_seed is None:
                train_seed = time.time_ns() & 0xFFFFFFFF if global_rank == 0 else 0
                train_seed = ppo.dist_sum(train_seed, train_args["device"])
            train_args["seed"] = train_seed
            torch_seed, env_seed = ppo.derive_rank_seeds(
                startup_args["vec"].get("seed"), train_seed, world_size, global_rank,
            )
            random.seed(torch_seed)
            np.random.seed(torch_seed % 2**32)
            torch.manual_seed(torch_seed)

            config = {key: value for key, value in train_args.items() if key in SAC_TRAIN_KEYS}
            config.update(
                env=env_name,
                run_name=run_name,
            )
            startup_args["run_name"] = run_name
            startup_args.setdefault("no_model_upload", False)
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

            learner_device = torch.device(ppo.torch_device(config["device"]))
            if policy is None:
                policy = load_policy(startup_args, vecenv, env_name)
            elif load_model_path is not None:
                _load_actor_weights(policy, load_model_path, learner_device, ppo)
            if critic is None:
                from pufferlib.ocean.torch_sac import DriveSACCritic
                critic = DriveSACCritic(vecenv.driver_env, **startup_args["policy"]).to(
                    device=learner_device, dtype=torch.float32,
                )
            if replay is None:
                from pufferlib.sac_replay import TensorReplay
                replay_device = "cpu" if config["cpu_offload"] else learner_device
                replay = TensorReplay(
                    config["batch_size"], vecenv.single_observation_space.shape,
                    device=replay_device,
                    pin_memory=config["cpu_offload"] and learner_device.type == "cuda",
                )
            config["action_count"] = policy.action_count

            startup_args["git"] = ppo._get_git_metadata()
            experiment_config = startup_args

            if logger is None and global_rank == 0:
                if startup_args.get("neptune"):
                    from pufferlib.sac_logging import SACNeptuneLogger

                    logger = SACNeptuneLogger(startup_args)
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
            load_model_path = args.get("load_model_path")
            previous_sigint_handler = signal.getsignal(signal.SIGINT)
            try:
                from pufferlib import pufferl as ppo
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)
            resume_state_path, auto_resume = _resolve_resume_state_path(
                config, load_model_path, args.get("load_id"),
            )
            config["resume_state_path"] = resume_state_path
            for key in (
                "load_model_path", "load_id", "wandb", "wandb_project", "wandb_group",
                "neptune", "neptune_name", "neptune_project", "no_model_upload", "tag",
            ):
                config.pop(key, None)
            if (load_model_path is not None or args.get("load_id") is not None) and policy is None:
                raise ValueError("Injected SAC actor warmstart requires an explicitly supplied policy")
            owns_process_group, config["device"], world_size, global_rank = _initialize_process_group(
                config.get("device", DEFAULT_CONFIG["device"]), ppo,
            )
            if global_rank != 0:
                logger = None
                cleanup_logger = None
            if args.get("load_id") is not None:
                load_model_path = _download_remote_model(args, global_rank, world_size)
            if load_model_path is not None:
                _load_actor_weights(policy, load_model_path, ppo.torch_device(config["device"]), ppo)

        pufferl = PuffeSAC(config, vecenv, policy, critic, replay, logger)
        cleanup_logger = pufferl.logger
        resume_state_path = pufferl.config["resume_state_path"]
        if resume_state_path is not None:
            pufferl.load_training_state(resume_state_path, same_run=auto_resume)
        if experiment_config is not None and global_rank == 0:
            ppo._save_experiment_config(experiment_config, pufferl.config["data_dir"])
        remaining_steps = max(0, pufferl.config["total_timesteps"] - pufferl.agent_steps)
        for _ in range(remaining_steps):
            if pufferl.agent_steps >= pufferl.config["total_timesteps"]:
                break
            pufferl.evaluate()
            logs = pufferl.train()
            if logs is not None:
                all_logs.append(logs)
            if (
                training_evaluation_scheduled
                and pufferl.epoch > 0
                and pufferl.epoch % startup_args["train"]["evaluation_interval_epochs"] == 0
                and pufferl.epoch != last_training_evaluation_epoch
                and global_rank == 0
            ):
                last_training_evaluation_epoch = pufferl.epoch
                ppo.run_training_evaluation(
                    env_name=env_name,
                    args=startup_args,
                    policy=pufferl.uncompiled_policy,
                    logger=pufferl.logger,
                    epoch=pufferl.epoch,
                    global_step=pufferl.agent_steps,
                    run_dir=config["data_dir"],
                )
            rank_zero_early_stop = (
                pufferl.rank == 0 and logs is not None
                and early_stop_fn is not None and early_stop_fn(logs)
            )
            if pufferl.distributed:
                rank_zero_early_stop = bool(
                    pufferl.helpers.dist_sum(int(rank_zero_early_stop), pufferl.device)
                )
            if rank_zero_early_stop:
                early_stop = True
                break
        if training_evaluation_scheduled and pufferl.epoch > 0:
            if global_rank == 0 and last_training_evaluation_epoch != pufferl.epoch:
                ppo.run_training_evaluation(
                    env_name=env_name,
                    args=startup_args,
                    policy=pufferl.uncompiled_policy,
                    logger=pufferl.logger,
                    epoch=pufferl.epoch,
                    global_step=pufferl.agent_steps,
                    run_dir=config["data_dir"],
                )
            if pufferl.distributed:
                torch.distributed.barrier()
        model_path = pufferl.close()
        close_started = True
        pufferl.logger.close(model_path, early_stop)
        return all_logs
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
    finally:
        if owns_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] not in {"train", "eval", "sweep", "controlled_exp", "profile", "autotune"}:
        raise NotImplementedError(
            "SAC commands: train, eval, sweep, controlled_exp, profile, autotune; "
            "follow the command with env_name and Hydra overrides"
        )
    command = argv[0]
    env_name = argv[1]
    overrides = argv[2:]
    benchmark_names = None
    if command == "eval" and overrides and "=" not in overrides[0] and not overrides[0].startswith(("+", "~", "--")):
        benchmark_names = overrides.pop(0)
    args = load_config(env_name, overrides=overrides)
    if command == "train":
        return train(env_name=env_name, args=args)
    if command == "eval":
        return eval(env_name=env_name, args=args, benchmark_names=benchmark_names)
    from pufferlib import sac_workflows
    if command == "autotune":
        return sac_workflows.autotune(args=args)
    return getattr(sac_workflows, command)(env_name=env_name, args=args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
