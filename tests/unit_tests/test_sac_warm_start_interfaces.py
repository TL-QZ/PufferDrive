from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pufferlib import pufferl, pufferl_sac
from pufferlib.pufferl_sac import FINAL_OBSERVATIONS_INFO_KEY
from pufferlib.sac_replay import TensorReplay


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, observations):
        return self.linear(observations)


class _TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Linear(2, 3)
        self.q2 = nn.Linear(2, 3)

    def forward(self, observations):
        return self.q1(observations), self.q2(observations)


class _VecEnv:
    single_observation_space = SimpleNamespace(shape=(2,))
    single_action_space = SimpleNamespace(n=3)

    def __init__(self):
        self.num_agents = 4
        self.agents_per_batch = 4
        self.reset_calls = 0

    def async_reset(self, seed):
        self.reset_calls += 1
        self.reset_seed = seed


class _NoopUtilization:
    def stop(self):
        pass

    def join(self):
        pass


class _NoopLogger:
    pass


class _Profile:
    def __call__(self, *args, **kwargs):
        return None

    def end(self):
        return None

    def clear(self):
        return None


def _make_initialized_learner(monkeypatch, tmp_path, *, reset_env=True, **config_updates):
    monkeypatch.setattr(pufferl, "Utilization", _NoopUtilization)
    monkeypatch.setattr(pufferl.PuffeRL, "print_dashboard", lambda self, clear=False: None)
    vecenv = _VecEnv()
    config = {
        "data_dir": str(tmp_path),
        "action_count": 3,
        "total_timesteps": 20,
        "batch_size": 4,
        "minibatch_size": 2,
        "max_minibatch_size": 2,
        "checkpoint_interval": None,
        **config_updates,
    }
    learner = pufferl_sac.PuffeSAC(
        config,
        vecenv,
        _TinyActor(),
        _TinyCritic(),
        TensorReplay(config["batch_size"], (2,)),
        logger=_NoopLogger(),
        reset_env=reset_env,
    )
    return learner, vecenv


def _make_update_learner(*, learn_alpha):
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.uncompiled_policy = _TinyActor()
    learner.policy = learner.uncompiled_policy
    learner.policy_train = learner.policy
    learner.ddp_policy = learner.policy
    learner.uncompiled_critic = _TinyCritic()
    learner.critic = learner.uncompiled_critic
    learner.critic_train = learner.critic
    learner.ddp_critic = learner.critic
    learner.uncompiled_target_critic = learner.uncompiled_critic.__class__()
    learner.uncompiled_target_critic.load_state_dict(learner.uncompiled_critic.state_dict())
    learner.uncompiled_target_critic.requires_grad_(False).eval()
    learner.target_critic = learner.uncompiled_target_critic
    learner.amp_context = nullcontext()
    learner.log_alpha = nn.Parameter(torch.tensor(float(np.log(0.1))))
    learner.optimizer = torch.optim.Adam(learner.policy.parameters(), lr=0.003)
    learner.critic_optimizer = torch.optim.Adam(learner.critic.parameters(), lr=0.004)
    learner.alpha_optimizer = torch.optim.Adam([learner.log_alpha], lr=0.002)
    learner.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(learner.optimizer, T_max=5)
    learner.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        learner.critic_optimizer, T_max=5,
    )
    learner.config = {
        "update_epochs": 1,
        "target_entropy": 0.7,
        "tau": 0.15,
        "max_grad_norm": None,
        "learn_alpha": learn_alpha,
        "anneal_lr": True,
        "total_timesteps": 20,
        "checkpoint_interval": None,
    }
    learner.batch_size = 4
    learner.minibatch_size = 4
    learner.max_minibatch_size = 4
    learner.replay = TensorReplay(4, (2,))
    observations = torch.tensor([[0.1, 0.2], [0.3, -0.4], [0.5, 0.2], [-0.2, 0.6]])
    learner.replay.add(
        observations,
        torch.tensor([0, 1, 2, 0]),
        torch.tensor([0.2, -0.1, 0.5, 0.3]),
        observations * 0.7 + 0.15,
        torch.tensor([0.99, 0.99, 0.0, 0.99]),
    )
    learner.distributed = False
    learner.world_size = 1
    learner.rank = 0
    learner.learner_updates = 0
    learner.sampled_transitions = 0
    learner.profile = _Profile()
    learner.epoch = 0
    learner.losses = {}
    learner.agent_steps = 0
    learner.global_step = 0
    learner.last_log_step = 0
    learner.last_log_time = time.time()
    learner.helpers = SimpleNamespace(dist_sum=lambda value, device: 0)
    return learner


def _module_state(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def test_constructor_can_keep_existing_environment_without_reset(monkeypatch, tmp_path):
    learner, vecenv = _make_initialized_learner(monkeypatch, tmp_path, reset_env=False)

    assert vecenv.reset_calls == 0
    assert not learner.pending.any()
    assert learner.config["learn_alpha"] is True


def test_old_version_five_checkpoint_without_learn_alpha_uses_learning_default(monkeypatch, tmp_path):
    learner, _ = _make_initialized_learner(monkeypatch, tmp_path)
    checkpoint = {
        "state_version": 5,
        "algorithm": "sac",
        "scope": "learner_only",
        "config": {key: value for key, value in learner.config.items() if key != "learn_alpha"},
        "policy_state_dict": learner.uncompiled_policy.state_dict(),
        "critic_state_dict": learner.uncompiled_critic.state_dict(),
        "target_critic_state_dict": learner.uncompiled_target_critic.state_dict(),
        "log_alpha": learner.log_alpha.detach().cpu(),
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "critic_optimizer_state_dict": learner.critic_optimizer.state_dict(),
        "alpha_optimizer_state_dict": learner.alpha_optimizer.state_dict(),
        "scheduler_state_dict": learner.scheduler.state_dict(),
        "critic_scheduler_state_dict": learner.critic_scheduler.state_dict(),
        "global_step": 0,
        "agent_steps": 0,
        "epoch": 0,
        "valid_inserts": 0,
        "learner_updates": 0,
        "sampled_transitions": 0,
        "best_score": -float("inf"),
        "rng_state": pufferl.capture_rng_state(),
    }
    checkpoint_path = tmp_path / "old_trainer_state.pt"
    torch.save(checkpoint, checkpoint_path)

    learner.load_training_state(checkpoint_path)

    assert learner.config["learn_alpha"] is True
    torch.testing.assert_close(learner.alpha, checkpoint["log_alpha"].exp())


def test_record_actions_copies_pending_state_by_agent_slot():
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.pending = torch.zeros(50, dtype=torch.bool)
    learner.observations = torch.zeros((50, 2), dtype=torch.float32)
    learner.actions = torch.zeros(50, dtype=torch.int64)
    observations = np.asarray([[4, 5], [6, 7]], dtype=np.float32)
    actions = np.asarray([2, 1], dtype=np.int64)

    learner.record_actions(observations, actions, np.asarray([42, 7], dtype=np.int64))
    observations[:] = -1
    actions[:] = 0

    np.testing.assert_array_equal(learner.observations[[42, 7]].numpy(), [[4, 5], [6, 7]])
    assert learner.actions[[42, 7]].tolist() == [2, 1]
    assert learner.pending[[42, 7]].tolist() == [True, True]
    assert not learner.pending[11]


def test_observe_transition_preserves_native_mask_terminal_and_truncation_rules():
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.config = {"gamma": 0.9}
    learner.pending = torch.zeros(50, dtype=torch.bool)
    learner.observations = torch.zeros((50, 2), dtype=torch.float32)
    learner.actions = torch.zeros(50, dtype=torch.int64)
    learner.record_actions(
        torch.tensor([[4, 5], [6, 7], [8, 9]]),
        torch.tensor([1, 2, 0]),
        torch.tensor([42, 7, 31]),
    )
    pending_before = learner.pending.clone()
    observations_before = learner.observations.clone()
    actions_before = learner.actions.clone()
    infos = [
        {
            FINAL_OBSERVATIONS_INFO_KEY: {
                "env_ids": np.asarray([42], dtype=np.int64),
                "observations": np.asarray([[9, 10]], dtype=np.float32),
            },
            "score": 3.0,
        },
        {"episode_length": 8},
    ]

    transitions, cleaned_infos = learner.observe_transition(
        np.asarray([[420, 421], [70, 71], [110, 111], [310, 311]], dtype=np.float32),
        np.asarray([2, 3, 4, 5], dtype=np.float32),
        np.asarray([False, True, False, False]),
        np.asarray([True, False, False, False]),
        infos,
        np.asarray([42, 7, 11, 31], dtype=np.int64),
        np.asarray([True, True, True, False]),
    )

    transition_observations, transition_actions, rewards, next_observations, discounts = transitions
    torch.testing.assert_close(transition_observations, torch.tensor([[4, 5], [6, 7]], dtype=torch.float32))
    assert transition_actions.tolist() == [1, 2]
    torch.testing.assert_close(rewards, torch.tensor([2, 3], dtype=torch.float32))
    torch.testing.assert_close(next_observations, torch.tensor([[9, 10], [6, 7]], dtype=torch.float32))
    torch.testing.assert_close(discounts, torch.tensor([0.9, 0.0]))
    assert cleaned_infos == [{"score": 3.0}, {"episode_length": 8}]
    assert FINAL_OBSERVATIONS_INFO_KEY not in cleaned_infos[0]
    torch.testing.assert_close(learner.pending, pending_before)
    torch.testing.assert_close(learner.observations, observations_before)
    torch.testing.assert_close(learner.actions, actions_before)

    learner.record_actions(torch.tensor([[0, 0], [1, 1]]), torch.tensor([0, 0]), torch.tensor([42, 7]))
    torch.testing.assert_close(transition_observations, torch.tensor([[4, 5], [6, 7]], dtype=torch.float32))
    torch.testing.assert_close(next_observations, torch.tensor([[9, 10], [6, 7]], dtype=torch.float32))


def test_observe_transition_rejects_missing_truncation_final_observation():
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.config = {"gamma": 0.9}
    learner.pending = torch.tensor([True])
    learner.observations = torch.zeros((1, 2), dtype=torch.float32)
    learner.actions = torch.zeros(1, dtype=torch.int64)

    with pytest.raises(ValueError, match="Missing pre-reset final observations"):
        learner.observe_transition(
            torch.ones((1, 2)), torch.ones(1), torch.zeros(1), torch.ones(1), [],
            torch.tensor([0]), torch.ones(1, dtype=torch.bool),
        )


def test_critic_only_update_uses_external_target_policy_and_skips_actor_alpha_steps():
    learner = _make_update_learner(learn_alpha=True)
    policy_before = _module_state(learner.policy)
    target_critic_before = _module_state(learner.target_critic)
    alpha_before = learner.log_alpha.detach().clone()
    actor_scheduler_state = learner.scheduler.state_dict()
    critic_scheduler_epoch = learner.critic_scheduler.last_epoch
    target_logits = nn.Parameter(torch.tensor([0.5, -0.2, 0.1]))
    target_policy_calls = []

    def target_policy(observations):
        target_policy_calls.append(observations.shape[0])
        return target_logits.expand(observations.shape[0], -1)

    replay_batch = learner.replay.get_batch(torch.arange(4))
    with torch.no_grad():
        next_q1, next_q2 = learner.target_critic(replay_batch[3])
        expected_logprobs = F.log_softmax(target_logits, dim=-1)
        expected_probs = expected_logprobs.exp()
        expected_next_value = (
            expected_probs * (torch.minimum(next_q1, next_q2) - learner.alpha * expected_logprobs)
        ).sum(-1)
        expected_targets = replay_batch[2] + replay_batch[4] * expected_next_value
    actual_targets = learner._compute_targets(
        replay_batch[2], replay_batch[3], replay_batch[4], target_policy=target_policy,
    )
    torch.testing.assert_close(actual_targets, expected_targets)
    assert not actual_targets.requires_grad

    target_policy_calls.clear()
    learner.train(update_policy=False, target_policy=target_policy)

    assert target_policy_calls == [4]
    assert target_logits.grad is None
    for before, after in zip(policy_before, learner.policy.parameters()):
        torch.testing.assert_close(before, after)
    assert not learner.optimizer.state
    assert not learner.alpha_optimizer.state
    torch.testing.assert_close(learner.log_alpha, alpha_before)
    assert learner.scheduler.state_dict() == actor_scheduler_state
    assert learner.critic_scheduler.last_epoch == critic_scheduler_epoch + 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_critic_before, learner.target_critic.parameters())
    )
    assert "policy_loss" not in learner.losses
    assert "alpha_loss" not in learner.losses
    assert learner.learner_updates == 1


def test_fixed_alpha_mode_keeps_actor_updates_and_temperature_constant():
    learner = _make_update_learner(learn_alpha=False)
    policy_before = _module_state(learner.policy)
    alpha_before = learner.log_alpha.detach().clone()

    learner.train()

    assert any(
        not torch.equal(before, after)
        for before, after in zip(policy_before, learner.policy.parameters())
    )
    torch.testing.assert_close(learner.log_alpha, alpha_before)
    assert not learner.alpha_optimizer.state
    assert "policy_loss" in learner.losses
    assert "alpha_loss" not in learner.losses
    assert learner.scheduler.last_epoch == 1
