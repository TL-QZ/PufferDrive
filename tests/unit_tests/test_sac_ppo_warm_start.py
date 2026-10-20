import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from pufferlib import pufferl
from pufferlib.pufferl_sac_ppo_warm_start import (
    IsolatedRNG,
    ReplayRefresh,
    WarmStartCoordinator,
)
from pufferlib.sac_replay import TensorReplay


class _ActorBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 5)

    def forward(self, observations, ego_dim):
        return torch.tanh(self.linear(observations[:, :ego_dim]))


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_backbone = _ActorBackbone()
        self.actor_head = nn.Linear(5, 2)
        self.ego_dim = 3
        self.atn_dim = [2]
        self.action_count = 2
        self.is_continuous = False

    def forward(self, observations):
        return self.actor_head(self.actor_backbone(observations, self.ego_dim))


class _FakePPO:
    def __init__(self, policy):
        self.config = {"device": "cpu", "checkpoint_interval": None}
        self.global_step = 0
        self.epoch = 0
        self.uncompiled_policy = policy
        self.logger = SimpleNamespace()
        self.transition_observer = None

    def evaluate(self):
        observations = torch.arange(12, dtype=torch.float32).reshape(4, 3) + self.global_step
        self.transition_observer(
            observations=observations,
            rewards=torch.ones(4),
            terminals=torch.zeros(4, dtype=torch.bool),
            truncations=torch.zeros(4, dtype=torch.bool),
            infos=[{} for _ in range(4)],
            env_ids=torch.arange(4),
            masks=torch.ones(4, dtype=torch.bool),
            actions=torch.tensor([0, 1, 0, 1]),
        )

    def train(self):
        self.global_step += 4
        self.epoch += 1


class _FakeSAC:
    def __init__(self, policy):
        self.device = torch.device("cpu")
        self.uncompiled_policy = policy
        self.replay = TensorReplay(4, (3,))
        self.batch_size = 4
        self.refresh_count = 2
        self.config = {"total_timesteps": 100, "checkpoint_interval": 10}
        self.global_step = 0
        self.agent_steps = 0
        self.valid_inserts = 0
        self.learner_updates = 0
        self.pending_actions = None
        self.reset_calls = 0
        self.last_update_policy = None

    def observe_transition(
        self, *, observations, rewards, terminals, truncations, infos, env_ids, masks,
    ):
        return (
            observations.clone(),
            torch.zeros(observations.shape[0], dtype=torch.long),
            rewards.clone(),
            observations + 0.25,
            torch.ones(observations.shape[0]),
        ), infos

    def record_actions(self, observations, actions, env_ids):
        self.pending_actions = (observations.clone(), actions.clone(), env_ids.clone())

    def train(self, *, update_policy, target_policy):
        self.last_update_policy = update_policy
        with torch.no_grad():
            logits = target_policy(torch.zeros((1, 3)))
        assert logits.shape == (1, 2)
        assert not logits.requires_grad
        random.random()
        np.random.random()
        torch.rand(1)
        self.learner_updates += 3


def _transitions(values):
    observations = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    return (
        observations,
        torch.zeros(observations.shape[0], dtype=torch.long),
        observations[:, 0].clone(),
        observations + 1,
        torch.ones(observations.shape[0]),
    )


def _full_replay(capacity):
    replay = TensorReplay(capacity, (1,))
    observations = torch.full((capacity, 1), -1.0)
    replay.add(
        observations,
        torch.zeros(capacity, dtype=torch.long),
        torch.full((capacity,), -1.0),
        observations.clone(),
        torch.ones(capacity),
    )
    return replay


def _assert_rng_state_equal(left, right):
    assert left["python"] == right["python"]
    left_numpy, right_numpy = left["numpy"], right["numpy"]
    assert left_numpy[0] == right_numpy[0]
    np.testing.assert_array_equal(left_numpy[1], right_numpy[1])
    assert left_numpy[2:] == right_numpy[2:]
    torch.testing.assert_close(left["torch"], right["torch"], rtol=0, atol=0)
    if "cuda" in left:
        assert len(left["cuda"]) == len(right["cuda"])
        for left_state, right_state in zip(left["cuda"], right["cuda"]):
            torch.testing.assert_close(left_state, right_state, rtol=0, atol=0)


def test_initial_fill_spans_cycles_and_refresh_starts_only_next_cycle():
    replay = TensorReplay(5, (1,))
    refresh = ReplayRefresh(replay, capacity=5, quota=2, generator=torch.Generator().manual_seed(4))

    refresh.begin_cycle(all_ranks_full=False)
    assert refresh.add(_transitions([0, 1, 2])) == 3
    assert refresh.finish_cycle() == 3

    refresh.begin_cycle(all_ranks_full=False)
    assert refresh.add(_transitions([10, 11, 12, 13])) == 2
    assert refresh.add(_transitions([20, 21])) == 0
    assert refresh.finish_cycle() == 2
    assert len(replay) == 5
    assert (replay.observations[:, 0] >= 0).all()

    refresh.begin_cycle(all_ranks_full=True)
    assert refresh.add(_transitions([30, 31])) == 2
    assert refresh.finish_cycle() == 2
    assert (replay.observations[:, 0] >= 30).sum().item() == 2


def test_refresh_with_quota_candidates_updates_every_selected_destination():
    replay = _full_replay(capacity=8)
    refresh = ReplayRefresh(replay, capacity=8, quota=3, generator=torch.Generator().manual_seed(19))
    refresh.begin_cycle(all_ranks_full=True)

    assert refresh.add(_transitions([40, 41, 42])) == 3
    assert refresh.finish_cycle() == 3
    assert (replay.observations[:, 0] >= 40).sum().item() == 3


def test_reservoir_refresh_samples_uniformly_across_full_cycle():
    selections = torch.zeros(8, dtype=torch.int32)
    for seed in range(240):
        replay = _full_replay(capacity=4)
        refresh = ReplayRefresh(
            replay,
            capacity=4,
            quota=4,
            generator=torch.Generator().manual_seed(seed),
        )
        refresh.begin_cycle(all_ranks_full=True)
        refresh.add(_transitions(range(8)))
        assert refresh.finish_cycle() == 4
        for candidate in replay.observations[:, 0].long().tolist():
            selections[candidate] += 1

    # Every candidate should occur in about half the four-row reservoir samples.
    assert torch.all((selections >= 85) & (selections <= 155)), selections.tolist()


def test_critic_cycle_isolates_rng_and_handoff_keeps_pending_state_and_step_axis():
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    ppo = _FakePPO(_TinyPolicy())
    sac = _FakeSAC(_TinyPolicy())
    sac_rng = IsolatedRNG(pufferl, seed=71)
    coordinator = WarmStartCoordinator(
        ppo,
        sac,
        handoff_steps=4,
        sac_steps=6,
        sac_rng=sac_rng,
        replay_generator=torch.Generator().manual_seed(8),
    )
    ppo.transition_observer = coordinator.observe_transition
    ppo_state_before = pufferl.capture_rng_state()
    ppo_actor_before = {
        name: parameter.detach().clone()
        for name, parameter in ppo.uncompiled_policy.named_parameters()
    }

    first_cycle = coordinator.run_ppo_cycle()
    _assert_rng_state_equal(ppo_state_before, pufferl.capture_rng_state())
    second_cycle = coordinator.run_ppo_cycle()

    assert first_cycle["cumulative_agent_steps"] == 4
    assert second_cycle["cumulative_agent_steps"] == 8
    assert first_cycle["replay_size"] == second_cycle["replay_size"] == 4
    assert (first_cycle["valid_replay_inserts"], second_cycle["valid_replay_inserts"]) == (4, 2)
    assert (first_cycle["critic_updates"], second_cycle["critic_updates"]) == (3, 3)
    assert sac.last_update_policy is False
    assert sac.agent_steps == 8
    for name, parameter in ppo.uncompiled_policy.named_parameters():
        torch.testing.assert_close(parameter, ppo_actor_before[name], rtol=0, atol=0)

    replay_object = sac.replay
    pending_actions = sac.pending_actions
    handoff = coordinator.handoff()

    assert handoff["boundary_agent_steps"] == 8
    assert coordinator.sac_phase_steps == 0
    assert sac.global_step == sac.agent_steps == 8
    assert sac.config["total_timesteps"] == 14
    assert sac.config["checkpoint_interval"] == 10
    assert sac.replay is replay_object
    assert sac.pending_actions is pending_actions
    assert sac.reset_calls == 0
