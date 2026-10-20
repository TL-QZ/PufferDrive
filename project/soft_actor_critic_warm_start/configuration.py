"""Load the dedicated PPO-to-SAC warm-start configuration."""

from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config/warm_start.yaml"


def resolve_config(run_dir, seed=None, config_path=None):
    """Load local settings, inject run-specific paths, and validate the Drive env."""
    from pufferlib.config_schema import ENV_SCHEMAS

    run_dir = Path(run_dir).expanduser().resolve()
    config_path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    config = OmegaConf.load(config_path)

    if seed is not None and seed < 0:
        raise ValueError("seed must be nonnegative")

    config.warm_start.run_dir = str(run_dir)
    config.warm_start.run_name = run_dir.name
    for phase in ("ppo", "sac"):
        phase_config = config[phase]
        if seed is not None:
            phase_config.train.seed = seed

        env_name = phase_config.env_name
        if env_name not in ENV_SCHEMAS:
            raise ValueError(f"No environment schema registered for {env_name!r}")
        phase_config.env = OmegaConf.merge(
            OmegaConf.structured(ENV_SCHEMAS[env_name]), phase_config.env,
        )

    config.provenance.local_config = str(config_path)
    resolved_config = OmegaConf.to_container(
        config, resolve=True, enum_to_str=True, throw_on_missing=True,
    )
    resolved_config.pop("_shared", None)
    return resolved_config


def smoke_config(config):
    """Explicit reduced CPU validation profile; never a performance experiment."""
    result = deepcopy(config)
    for args in (result["ppo"], result["sac"]):
        args.update(wandb=False, neptune=False, tb=False)
        args["vec"].update(backend="Serial", num_envs=1, num_workers=1, batch_size=1)
        args["env"].update(num_agents=8, min_agents_per_env=8, max_agents_per_env=8, num_maps=1)
        args["train"].update(
            device="cpu", compile=False, amp=False, precision="float32",
            minibatch_size=32, max_minibatch_size=16, checkpoint_interval=None,
            evaluation_interval_epochs=None,
        )
        args["policy"].update(
            ego_input_size=8, partner_input_size=8, lane_input_size=8,
            boundary_input_size=8, traffic_control_input_size=8, context_input_size=8,
            backbone_hidden_size=16, backbone_num_layers=1, actor_hidden_size=8,
            critic_hidden_size=8,
        )
    # Keep the tiny rollout in one PPO minibatch: native advantage std is undefined
    # for a one-row partial minibatch after filtering.
    result["ppo"]["train"].update(
        batch_size="auto", bptt_horizon=8, total_timesteps=4096,
        minibatch_size=64, max_minibatch_size=64,
    )
    result["sac"]["train"].update(batch_size=32, total_timesteps=64)
    result["warm_start"].update(handoff_steps=128, sac_steps=64)
    result["provenance"].update(expected_world_size=1, smoke=True)
    return result


def memory_estimate(config, world_size=2):
    from project.soft_actor_critic_tuning.scripts.run_sac16_grid import (
        _estimate_observation_size,
    )

    features = _estimate_observation_size(config["ppo"]["env"])
    capacity = config["sac"]["train"]["batch_size"]
    return {
        "observation_features": features,
        "replay_bytes_all_ranks": world_size * capacity * (8 * features + 16),
        "note": "Replay only; excludes PPO storage, models, simulator, and working memory.",
    }
