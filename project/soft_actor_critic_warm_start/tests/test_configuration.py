import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError

from project.soft_actor_critic_warm_start.configuration import (
    DEFAULT_CONFIG_PATH,
    memory_estimate,
    resolve_config,
    smoke_config,
)


def _write_config(tmp_path, updates):
    config = OmegaConf.load(DEFAULT_CONFIG_PATH)
    for key, value in updates.items():
        OmegaConf.update(config, key, value)
    config_path = tmp_path / "custom_warm_start.yaml"
    OmegaConf.save(config, config_path)
    return config_path


def test_resolved_baseline_and_selected_trial_are_compatible(tmp_path):
    config = resolve_config(tmp_path / "run")
    ppo, sac = config["ppo"], config["sac"]
    assert ppo["policy"] == sac["policy"]
    assert ppo["env"] == sac["env"]
    assert ppo["train"]["total_timesteps"] == 10_000_000_000
    assert ppo["train"]["learning_rate"] == 0.0005
    assert ppo["train"]["anneal_lr"] is True
    assert ppo["train"]["update_epochs"] == sac["train"]["update_epochs"] == 3
    assert ppo["vec"] == {
        "backend": "Multiprocessing", "batch_size": 10, "num_envs": 20,
        "num_workers": "auto", "seed": 42, "zero_copy": True,
    }
    assert sac["vec"]["overwork"] is False
    assert ppo["eval"] == sac["eval"]
    assert ppo["eval"]["benchmark_config"] == (
        "project/baseline_run_sync_2026-08-24/override_config/evaluation_benchmarks.yaml"
    )
    assert sac["train"]["batch_size"] == 8_192_000
    assert sac["train"]["critic_learning_rate"] == 0.0005
    assert sac["train"]["gamma"] == 0.999
    assert sac["train"]["tau"] == 0.005
    assert sac["train"]["replay_update_ratio"] == 0.1
    assert sac["train"]["target_entropy"] == 1.863680
    assert sac["train"]["learn_alpha"] is False
    assert sac["train"]["initial_alpha"] == 0.1
    assert config["warm_start"]["handoff_steps"] == 2_000_000_000
    assert config["warm_start"]["sac_steps"] == 2_000_000_000
    assert sac["train"]["total_timesteps"] == 2_000_000_000
    assert ppo["train"]["data_dir"] != sac["train"]["data_dir"]
    assert config["provenance"]["local_config"] == str(DEFAULT_CONFIG_PATH.resolve())
    assert memory_estimate(config)["replay_bytes_all_ranks"] == 121_241_600_000


def test_custom_yaml_edits_control_handoff_budget_sac_budget_and_critic_rate(tmp_path):
    config_path = _write_config(tmp_path, {
        "warm_start.handoff_steps": 12_345,
        "warm_start.sac_steps": 54_321,
        "sac.train.critic_learning_rate": 0.00017,
        "_shared.policy.critic_hidden_size": 384,
    })
    config = resolve_config(tmp_path / "run", config_path=config_path)

    assert config["warm_start"]["handoff_steps"] == 12_345
    assert config["warm_start"]["sac_steps"] == 54_321
    assert config["sac"]["train"]["total_timesteps"] == 54_321
    assert config["sac"]["train"]["critic_learning_rate"] == 0.00017
    assert config["ppo"]["policy"]["critic_hidden_size"] == 384
    assert config["sac"]["policy"]["critic_hidden_size"] == 384


def test_yaml_seed_is_default_and_explicit_seed_overrides_it(tmp_path):
    default_config = resolve_config(tmp_path / "default-seed")
    assert default_config["ppo"]["train"]["seed"] == 0
    assert default_config["sac"]["train"]["seed"] == 0

    config_path = _write_config(tmp_path, {
        "ppo.train.seed": 7,
        "sac.train.seed": 7,
    })
    yaml_seed_config = resolve_config(tmp_path / "yaml-seed", config_path=config_path)
    override_config = resolve_config(tmp_path / "override-seed", seed=9, config_path=config_path)

    assert yaml_seed_config["ppo"]["train"]["seed"] == 7
    assert yaml_seed_config["sac"]["train"]["seed"] == 7
    assert override_config["ppo"]["train"]["seed"] == 9
    assert override_config["sac"]["train"]["seed"] == 9


def test_malformed_environment_field_fails_during_config_validation(tmp_path):
    config_path = _write_config(tmp_path, {"_shared.env.dt": "not-a-number"})

    with pytest.raises(ValidationError, match="could not be converted to Float"):
        resolve_config(tmp_path / "run", config_path=config_path)


def test_smoke_is_isolated_and_keeps_native_update_counts(tmp_path):
    full = resolve_config(tmp_path / "run", seed=2)
    smoke = smoke_config(full)
    assert full["sac"]["train"]["batch_size"] == 8_192_000
    assert smoke["ppo"]["policy"] == smoke["sac"]["policy"]
    assert smoke["ppo"]["env"] == smoke["sac"]["env"]
    for phase in ("ppo", "sac"):
        assert smoke[phase]["train"]["seed"] == 2
        assert smoke[phase]["train"]["update_epochs"] == 3
        assert smoke[phase]["train"]["device"] == "cpu"
        assert smoke[phase]["wandb"] is False
    assert smoke["provenance"]["expected_world_size"] == 1
