import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pufferlib import pufferl, pufferl_sac


def test_sac_config_inherits_drive_sections_and_filters_ppo_train_settings():
    args = pufferl_sac.load_config("puffer_drive", overrides=[])

    assert args["env_name"] == "puffer_drive"
    assert args["policy_name"] == "DriveSAC"
    assert args["env"]["action_type"] == "continuous"
    assert args["policy"]["action_type"] == "discrete"
    assert "eval" in args and "vec" in args
    assert args["train"]["batch_size"] == 10_000
    assert args["train"]["minibatch_size"] == 256
    assert args["train"]["max_minibatch_size"] == 256
    assert args["train"]["learning_starts"] == 10_000
    assert args["train"]["learning_rate"] == 3e-4
    assert args["train"]["critic_learning_rate"] == 3e-4
    assert args["train"]["alpha_learning_rate"] == 3e-4
    assert args["train"]["optimizer"] == "adam"
    assert args["train"]["initial_alpha"] == 0.2
    assert args["train"]["tau"] == 0.005
    assert not ({"compile", "amp", "precision", "clip_coef", "gae_lambda"} & args["train"].keys())


def test_sac_config_rejects_ppo_only_train_override():
    with pytest.raises(pufferl.pufferlib.APIUsageError, match="PPO-only train settings"):
        pufferl_sac.load_config("puffer_drive", overrides=["train.clip_coef=0.1"])


def test_main_passes_hydra_composed_config_to_train(monkeypatch):
    received = {}

    def fake_train(env_name, args):
        received["env_name"] = env_name
        received["args"] = args
        return "started"

    monkeypatch.setattr(pufferl_sac, "train", fake_train)
    result = pufferl_sac.main([
        "train", "puffer_drive", "train.total_timesteps=5", "env.num_agents=4",
    ])

    assert result == "started"
    assert received["env_name"] == "puffer_drive"
    assert received["args"]["train"]["total_timesteps"] == 5
    assert received["args"]["env"]["num_agents"] == 4


def _startup_args(tmp_path):
    args = copy.deepcopy(pufferl_sac.load_config("puffer_drive", overrides=[]))
    args["train"].update(total_timesteps=1, batch_size=2, learning_starts=1, data_dir=str(tmp_path))
    args["vec"].update(seed=7)
    args["wandb"] = False
    args["neptune"] = False
    args["tb"] = False
    return args


class _FakeVecEnv:
    def __init__(self, capture_final_observations=True):
        self.driver_env = SimpleNamespace(capture_final_observations=capture_final_observations)
        self.single_observation_space = SimpleNamespace(shape=(4,))
        self.agents_per_batch = 4
        self.num_agents = 4
        self.closed = False

    def close(self):
        self.closed = True


class _FakeLogger:
    def close(self, model_path, early_stop):
        self.closed = True


class _FakeLearner:
    def __init__(self, config, vecenv, policy, critic, replay, logger):
        self.config = {**pufferl_sac.DEFAULT_CONFIG, **config}
        self.global_step = config["total_timesteps"]
        self.closed = False
        self.logger = logger or _FakeLogger()
        self.vecenv = vecenv
        self.policy = policy
        self.critic = critic
        self.replay = replay

    def close(self):
        self.closed = True
        self.vecenv.close()
        return "final_model.pt"


def test_startup_seeds_before_models_and_builds_sac_components(monkeypatch, tmp_path):
    args = _startup_args(tmp_path)
    vecenv = _FakeVecEnv()
    actor = SimpleNamespace(action_count=12)
    critic = object()
    observed = {}

    def fake_load_env(env_name, startup_args, seed=None):
        observed["env_name"] = env_name
        observed["env_seed"] = seed
        observed["capture"] = startup_args["env"]["capture_final_observations"]
        return vecenv

    def fake_load_policy(startup_args, loaded_vecenv, env_name):
        observed["torch_seed"] = torch.initial_seed()
        observed["python_seed"] = random.getstate()
        observed["numpy_seed"] = np.random.get_state()[1][0]
        assert loaded_vecenv is vecenv
        return actor

    saved = {}
    learners = []

    def fake_learner(*learner_args):
        learner = _FakeLearner(*learner_args)
        learners.append(learner)
        return learner

    class FakeCritic:
        def __init__(self, driver_env, **policy_args):
            observed["critic_constructed"] = driver_env is vecenv.driver_env

        def to(self, device, dtype):
            observed["critic_device"] = device
            observed["critic_dtype"] = dtype
            return critic

    monkeypatch.setattr(pufferl, "load_env", fake_load_env)
    monkeypatch.setattr(pufferl, "_get_git_metadata", lambda: {"commit_hash": "test"})
    monkeypatch.setattr(
        pufferl, "_save_experiment_config",
        lambda saved_args, path: saved.update(args=saved_args, path=path),
    )
    monkeypatch.setattr(pufferl_sac, "load_policy", fake_load_policy)
    monkeypatch.setattr("pufferlib.ocean.torch_sac.DriveSACCritic", FakeCritic)
    monkeypatch.setattr(pufferl_sac, "PuffeSAC", fake_learner)

    result = pufferl_sac.train(args=args)
    learner = learners[0]

    expected_torch_seed, expected_env_seed = pufferl.derive_rank_seeds(7, 42, 1, 0)
    assert observed["torch_seed"] == expected_torch_seed
    assert observed["python_seed"] == random.Random(expected_torch_seed).getstate()
    assert observed["numpy_seed"] == expected_torch_seed
    assert observed["env_seed"] == expected_env_seed
    assert observed["capture"] is True
    assert observed["critic_constructed"] is True
    assert observed["critic_device"] == "cpu"
    assert observed["critic_dtype"] == torch.float32
    assert learner.config["action_count"] == actor.action_count
    assert learner.config["data_dir"] == str(tmp_path)
    assert "clip_coef" not in learner.config and "compile" not in learner.config
    assert saved["path"] == str(tmp_path)
    assert saved["args"]["env"]["capture_final_observations"] is True
    assert saved["args"]["git"] == {"commit_hash": "test"}
    assert learner.closed and learner.logger.closed
    assert result == {}


def test_startup_requires_final_observation_transport(monkeypatch, tmp_path):
    args = _startup_args(tmp_path)
    vecenv = _FakeVecEnv(capture_final_observations=False)
    monkeypatch.setattr(pufferl, "load_env", lambda *args, **kwargs: vecenv)
    monkeypatch.setattr(
        pufferl_sac, "load_policy",
        lambda *args, **kwargs: pytest.fail("policy must not be built without final-observation transport"),
    )

    with pytest.raises(NotImplementedError, match="final-observation transport"):
        pufferl_sac.train(args=args)
    assert vecenv.closed


def test_startup_resolves_auto_replay_sizes_before_allocation(monkeypatch, tmp_path):
    args = _startup_args(tmp_path)
    args["train"].update(
        batch_size="auto", minibatch_size="auto", learning_starts=3, max_minibatch_size=2,
    )
    vecenv = _FakeVecEnv()
    policy = SimpleNamespace(action_count=12)
    observed = {}

    class FakeReplay:
        def __init__(self, capacity, observation_shape, device):
            observed["replay_capacity"] = capacity

    class FakeLearner(_FakeLearner):
        def __init__(self, config, vecenv, policy, critic, replay, logger):
            super().__init__(config, vecenv, policy, critic, replay, logger)
            observed["learner_capacity"] = config["batch_size"]
            observed["learner_minibatch_size"] = config["minibatch_size"]

    saved = {}
    monkeypatch.setattr("pufferlib.sac_replay.TensorReplay", FakeReplay)
    monkeypatch.setattr(pufferl, "_get_git_metadata", lambda: {})
    monkeypatch.setattr(
        pufferl, "_save_experiment_config",
        lambda saved_args, path: saved.update(train=dict(saved_args["train"])),
    )
    monkeypatch.setattr(pufferl_sac, "PuffeSAC", FakeLearner)

    pufferl_sac.train(
        args=args, vecenv=vecenv, policy=policy, critic=object(), logger=_FakeLogger(),
    )

    assert observed["replay_capacity"] == 256
    assert observed["learner_capacity"] == 256
    assert observed["learner_minibatch_size"] == 256
    assert saved["train"]["batch_size"] == 256
    assert saved["train"]["minibatch_size"] == 256


def test_flat_injected_train_api_remains_available(monkeypatch):
    config = {
        "total_timesteps": 1,
        "batch_size": 1,
        "minibatch_size": 1,
        "update_epochs": 1,
        "learning_starts": 1,
        "action_count": 12,
        "data_dir": "unused",
    }
    vecenv = _FakeVecEnv()
    monkeypatch.setattr(pufferl_sac, "PuffeSAC", _FakeLearner)

    result = pufferl_sac.train(
        args=config,
        vecenv=vecenv,
        policy=SimpleNamespace(action_count=12),
        critic=object(),
        replay=object(),
    )

    assert result == {}
    assert vecenv.closed
