"""Tests for the optional PPO collection observer."""

from collections import defaultdict
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pufferlib import pufferl
from pufferlib.spaces import Box, Discrete


class _Profile:
    def __call__(self, *_args, **_kwargs):
        return None

    def end(self):
        return None


class _Utilization:
    def __init__(self):
        pass


class _Policy(torch.nn.Module):
    is_continuous = False

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 10.0]))

    def forward_eval(self, observations, _state):
        logits = self.logits.expand(observations.shape[0], -1)
        values = 10.0 + observations[:, 0]
        return logits, values


class _VecEnv:
    def __init__(self, *, truncations=False, include_final_observations=False):
        self.step = 0
        self.sent_actions = []
        self.truncations = truncations
        self.include_final_observations = include_final_observations

    def recv(self):
        step = self.step
        self.step += 1
        info = {"episode": {"score": float(step)}}
        if self.include_final_observations:
            info["_pufferlib_final_observations"] = {
                "env_ids": np.asarray([0], dtype=np.int64),
                "observations": np.asarray([[100.0 + step, 200.0 + step]], dtype=np.float32),
            }

        is_truncated = self.truncations and step in (1, 3)
        return (
            np.asarray([[float(step), float(step + 1)]], dtype=np.float32),
            np.asarray([2.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([float(is_truncated)], dtype=np.float32),
            [info],
            np.asarray([0], dtype=np.int64),
            np.asarray([True]),
        )

    def send(self, actions):
        self.sent_actions.append(np.asarray(actions).copy())


def _make_collector(observer=None, *, truncations=False, include_final_observations=False):
    learner = pufferl.PuffeRL.__new__(pufferl.PuffeRL)
    learner.transition_observer = observer
    learner.profile = _Profile()
    learner.epoch = 0
    learner.config = {
        "device": "cpu",
        "use_rnn": False,
        "normalize_rewards": truncations,
        "bptt_horizon": 2,
        "cpu_offload": False,
        "use_value_bootstrapping": truncations,
        "gamma": 0.9,
    }
    learner.vecenv = _VecEnv(
        truncations=truncations,
        include_final_observations=include_final_observations,
    )
    learner.env_continuous = False
    learner.uncompiled_policy = _Policy()
    learner.policy = learner.uncompiled_policy
    learner.global_step = 0
    learner.segments = 2
    learner.total_agents = 1
    learner.full_rows = 0
    learner.observations = torch.zeros(2, 2, 2)
    learner.actions = torch.zeros(2, 2, dtype=torch.int64)
    learner.values = torch.zeros(2, 2)
    learner.logprobs = torch.zeros(2, 2)
    learner.rewards = torch.zeros(2, 2)
    learner.terminals = torch.zeros(2, 2)
    learner.truncations = torch.zeros(2, 2)
    learner.masks = torch.zeros(2, 2, dtype=torch.bool)
    learner.ep_lengths = torch.zeros(1, dtype=torch.int32)
    learner.ep_indices = torch.arange(1, dtype=torch.int32)
    learner.free_idx = 1
    learner.stats = defaultdict(list)
    learner.amp_context = torch.autocast(device_type="cpu", enabled=False)
    return learner


def _run_collector(learner):
    torch.manual_seed(2026)
    metrics = learner.evaluate()
    return {
        "metrics": metrics,
        "observations": learner.observations.clone(),
        "actions": learner.actions.clone(),
        "values": learner.values.clone(),
        "logprobs": learner.logprobs.clone(),
        "rewards": learner.rewards.clone(),
        "terminals": learner.terminals.clone(),
        "truncations": learner.truncations.clone(),
        "masks": learner.masks.clone(),
        "sent_actions": learner.vecenv.sent_actions,
        "random_state": torch.random.get_rng_state().clone(),
    }


def test_observation_only_observer_preserves_native_ppo_collection():
    native = _run_collector(_make_collector())
    observed_observations = []

    def observe(**transition):
        observed_observations.append(transition["observations"].clone())
        return transition["infos"]

    observed = _run_collector(_make_collector(observe))

    assert len(observed_observations) == 4
    for observation in observed_observations:
        assert observation.shape == (1, 2)
    for key in native:
        if key == "metrics":
            assert observed[key] == native[key]
        elif key == "sent_actions":
            for observed_actions, native_actions in zip(observed[key], native[key]):
                np.testing.assert_array_equal(observed_actions, native_actions)
        else:
            assert torch.equal(observed[key], native[key])


def test_observer_sees_raw_transition_and_its_cleaned_infos_are_aggregated():
    captured = []

    def observe(**transition):
        assert not torch.is_grad_enabled()
        captured.append({
            "observations": transition["observations"].clone(),
            "rewards": transition["rewards"].clone(),
            "terminals": transition["terminals"].clone(),
            "truncations": transition["truncations"].clone(),
            "env_ids": transition["env_ids"].clone(),
            "masks": np.asarray(transition["masks"]).copy(),
            "actions": transition["actions"].clone(),
            "infos": transition["infos"],
            "sent_count": len(learner.vecenv.sent_actions),
        })
        return [
            {
                "episode": info["episode"],
                "observer": {"raw_reward": float(transition["rewards"][0].item())},
            }
            for info in transition["infos"]
        ]

    learner = _make_collector(
        observe,
        truncations=True,
        include_final_observations=True,
    )
    metrics = learner.evaluate()

    assert len(captured) == 4
    for step, transition in enumerate(captured):
        assert transition["observations"].tolist() == [[float(step), float(step + 1)]]
        assert transition["rewards"].tolist() == [2.0]
        assert transition["env_ids"].tolist() == [0]
        assert transition["masks"].tolist() == [True]
        assert "_pufferlib_final_observations" in transition["infos"][0]
        assert transition["actions"].item() == learner.actions[step // 2, step % 2].item()
        assert transition["sent_count"] == step
        assert transition["terminals"].tolist() == [0.0]
        assert transition["truncations"].item() == float(step in (1, 3))

    normalized_reward = np.log1p(2.0)
    assert learner.rewards[0, 0].item() == pytest.approx(normalized_reward)
    assert learner.rewards[0, 1].item() == pytest.approx(normalized_reward + 0.9 * 10.0)
    assert learner.rewards[1, 1].item() == pytest.approx(normalized_reward + 0.9 * 12.0)
    assert metrics["episode/score"] == pytest.approx(1.5)
    assert metrics["observer/raw_reward"] == pytest.approx(2.0)
    assert all("_pufferlib_final_observations" not in key for key in metrics)


def test_transition_observer_is_an_optional_constructor_keyword_and_is_stored(monkeypatch):
    observer_parameter = inspect.signature(pufferl.PuffeRL.__init__).parameters["transition_observer"]
    assert observer_parameter.default is None

    monkeypatch.setattr(pufferl.PuffeRL, "print_dashboard", lambda self, clear=False: None)
    monkeypatch.setattr(pufferl, "Utilization", _Utilization)

    class _InitVecEnv:
        num_agents = 1
        single_observation_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        single_action_space = Discrete(2)

        def async_reset(self, seed):
            self.seed = seed

    observer = lambda **transition: transition["infos"]
    config = {
        "torch_deterministic": False,
        "seed": 0,
        "batch_size": 2,
        "bptt_horizon": 1,
        "device": "cpu",
        "precision": "float32",
        "cpu_offload": False,
        "use_rnn": False,
        "render": False,
        "render_interval": 1,
        "minibatch_size": 1,
        "max_minibatch_size": 1,
        "update_epochs": 1,
        "compile": False,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "adam_weight_decay": 0.0,
        "total_timesteps": 2,
        "amp": False,
    }
    learner = pufferl.PuffeRL(
        config,
        _InitVecEnv(),
        _Policy(),
        logger=SimpleNamespace(),
        transition_observer=observer,
    )

    assert learner.transition_observer is observer
