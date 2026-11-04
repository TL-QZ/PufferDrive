"""Collector and replay integration checks for Peng Q(lambda) discrete SAC."""

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pufferlib import pufferl_sac
from pufferlib.sac_sequence_replay import SequenceReplay


class _Profile:
    def __call__(self, *args, **kwargs):
        return None

    def end(self):
        return None


class _ScriptedVecEnv:
    single_observation_space = SimpleNamespace(shape=(1,))

    def __init__(self, num_agents, calls):
        self.num_agents = num_agents
        self.calls = calls
        self.receive_count = 0
        self.sent_actions = []

    def recv(self):
        call = self.calls[self.receive_count]
        self.receive_count += 1
        infos = []
        if "final_observations" in call:
            infos = [{pufferl_sac.FINAL_OBSERVATIONS_INFO_KEY: {
                "env_ids": np.asarray(call["final_ids"], dtype=np.int64),
                "observations": np.asarray(call["final_observations"], dtype=np.float32).reshape(-1, 1),
            }}]
        return (
            np.asarray(call["observations"], dtype=np.float32).reshape(-1, 1),
            np.asarray(call.get("rewards", np.zeros(len(call["ids"]))), dtype=np.float32),
            np.asarray(call.get("terminals", np.zeros(len(call["ids"]))), dtype=np.bool_),
            np.asarray(call.get("truncations", np.zeros(len(call["ids"]))), dtype=np.bool_),
            infos,
            np.asarray(call["ids"], dtype=np.int64),
            np.asarray(call.get("masks", np.ones(len(call["ids"]))), dtype=np.bool_),
        )

    def send(self, actions):
        self.sent_actions.append(np.asarray(actions).copy())


def _call(ids, observations, *, rewards=None, terminals=None, truncations=None,
          masks=None, final_ids=None, final_observations=None):
    result = {"ids": ids, "observations": observations}
    for name, value in (
        ("rewards", rewards),
        ("terminals", terminals),
        ("truncations", truncations),
        ("masks", masks),
        ("final_ids", final_ids),
        ("final_observations", final_observations),
    ):
        if value is not None:
            result[name] = value
    return result


def _make_pql_collector(vecenv, *, capacity, total_timesteps, replay_update_ratio=1.0):
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.profile = _Profile()
    learner.epoch = 0
    learner.replay = SequenceReplay(capacity, (1,), vecenv.num_agents)
    learner.batch_size = capacity
    learner.refresh_count = max(1, int(np.ceil(capacity * replay_update_ratio)))
    learner.config = {
        "total_timesteps": total_timesteps,
        "gamma": 0.9,
        "n_steps": 1,
        "return_estimator": "pql",
        "pql_horizon": 8,
        "pql_lambda": 0.95,
        "pql_entropy_convention": "soft_each_return_endpoint",
    }
    learner.global_step = 0
    learner.agent_steps = 0
    learner.world_size = 1
    learner.rank = 0
    learner.distributed = False
    learner.valid_inserts = 0
    learner.device = torch.device("cpu")
    learner.vecenv = vecenv
    learner.pending = torch.zeros(vecenv.num_agents, dtype=torch.bool)
    learner.observations = torch.zeros((vecenv.num_agents, 1), dtype=torch.float32)
    learner.actions = torch.zeros(vecenv.num_agents, dtype=torch.int64)
    learner.stats = defaultdict(list)
    learner.env_continuous = False
    learner.policy = lambda observations: torch.zeros((observations.shape[0], 2))
    learner.uncompiled_policy = learner.policy
    learner.amp_context = nullcontext()
    learner.pytorch = SimpleNamespace(
        sample_logits=lambda logits, **kwargs: (
            torch.zeros(logits.shape[0], dtype=torch.long), None, None, None,
        )
    )
    learner.helpers = SimpleNamespace(
        pufferlib=SimpleNamespace(
            unroll_nested_dict=lambda info: [],
            utils=SimpleNamespace(reduce_environment_metrics=lambda stats: {}),
        )
    )
    return learner


def _planned_segments(replay, *, horizon=16, logical_batch_size=16):
    segments = []
    planned_indices = []
    for indices, offsets in replay.build_epoch_batches(horizon, logical_batch_size):
        planned_indices.extend(indices.tolist())
        observations = replay.get_batch(indices)[0].reshape(-1).tolist()
        metadata = replay.get_sequence_metadata(indices)
        for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
            segment_agent_ids = metadata["agent_ids"][start:end]
            assert torch.all(segment_agent_ids == segment_agent_ids[0])
            segments.append((int(segment_agent_ids[0]), observations[start:end]))
    assert sorted(planned_indices) == sorted(
        torch.nonzero(replay._valid, as_tuple=False).flatten().tolist()
    )
    assert len(planned_indices) == len(set(planned_indices)) == len(replay)
    return segments


def test_collector_keeps_stable_id_chains_separate_across_an_invalid_gap():
    vecenv = _ScriptedVecEnv(5, [
        _call([4, 1], [40, 10]),
        _call([1, 4], [11, 41], rewards=[1, 10]),
        _call([4, 1], [42, 12], rewards=[20, 2]),
        _call([1, 4], [13, 43], rewards=[3, 30], masks=[True, False]),
        _call([4, 1], [44, 14], rewards=[40, 4]),
    ])
    learner = _make_pql_collector(vecenv, capacity=8, total_timesteps=10)

    learner.evaluate()

    assert vecenv.receive_count == 5
    assert learner.valid_inserts == 7
    segments = _planned_segments(learner.replay)
    assert sorted(rows for agent_id, rows in segments if agent_id == 1) == [[10.0, 11.0, 12.0, 13.0]]
    assert sorted(rows for agent_id, rows in segments if agent_id == 4) == [[40.0, 41.0], [43.0]]
    assert learner.replay.last_plan_metrics["cut_gap_count"] == 1


def test_collector_distinguishes_terminal_and_timeout_and_uses_final_observation():
    vecenv = _ScriptedVecEnv(1, [
        _call([0], [1]),
        _call([0], [2], rewards=[2]),
        _call([0], [3], rewards=[3], terminals=[True], truncations=[True]),
        _call([0], [4], rewards=[4]),
        _call(
            [0], [5], rewards=[5], truncations=[True],
            final_ids=[0], final_observations=[99],
        ),
    ])
    learner = _make_pql_collector(vecenv, capacity=8, total_timesteps=5)

    learner.evaluate()

    assert vecenv.receive_count == 5
    assert learner.valid_inserts == 4
    rows = learner.replay.get_batch(torch.arange(4))
    metadata = learner.replay.get_sequence_metadata(torch.arange(4))
    torch.testing.assert_close(rows[0].reshape(-1), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(rows[2], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    torch.testing.assert_close(rows[3].reshape(-1), torch.tensor([2.0, 2.0, 4.0, 99.0]))
    torch.testing.assert_close(rows[4], torch.tensor([0.9, 0.0, 0.9, 0.9]))
    assert metadata["terminals"].tolist() == [False, True, False, False]
    assert metadata["truncations"].tolist() == [False, False, False, True]
    assert sorted(rows for _agent, rows in _planned_segments(learner.replay)) == [
        [1.0, 2.0], [3.0, 4.0],
    ]
    assert learner.replay.last_plan_metrics["cut_terminal_count"] == 1
    assert learner.replay.last_plan_metrics["cut_truncation_count"] == 1


def test_collector_continues_pending_action_across_evaluate_calls_and_ring_wrap():
    vecenv = _ScriptedVecEnv(2, [
        _call([0, 1], [0, 1]),
        _call([0, 1], [10, 11], rewards=[2, 3]),
        _call([1], [21], rewards=[30]),
    ])
    learner = _make_pql_collector(
        vecenv, capacity=2, total_timesteps=5, replay_update_ratio=0.5,
    )

    learner.evaluate()
    assert vecenv.receive_count == 2
    assert learner.valid_inserts == 2
    assert learner.pending.tolist() == [True, True]

    learner.evaluate()

    assert vecenv.receive_count == 3
    assert learner.valid_inserts == 3
    segments = _planned_segments(learner.replay, horizon=4, logical_batch_size=2)
    assert segments == [(1, [1.0, 11.0])]
    rows = learner.replay.get_batch(torch.tensor([0, 1]))
    assert sorted(rows[2].tolist()) == [3.0, 30.0]


def test_quota_discard_breaks_the_dropped_agents_chain_before_a_later_insert(monkeypatch):
    vecenv = _ScriptedVecEnv(3, [
        _call([0, 1, 2], [0, 1, 2]),
        _call([0, 1, 2], [10, 11, 12], rewards=[10, 11, 12]),
        _call([0, 1, 2], [20, 21, 22], rewards=[20, 21, 22]),
        _call([2], [32], rewards=[32]),
    ])
    learner = _make_pql_collector(
        vecenv, capacity=5, total_timesteps=10, replay_update_ratio=0.2,
    )
    monkeypatch.setattr(torch, "randperm", lambda size, **kwargs: torch.arange(size))

    learner.evaluate()
    assert vecenv.receive_count == 3
    assert learner.valid_inserts == 5
    assert len(learner.replay) == 5

    learner.evaluate()

    assert vecenv.receive_count == 4
    assert learner.valid_inserts == 6
    segments = _planned_segments(learner.replay, horizon=8, logical_batch_size=8)
    assert sorted(rows for agent_id, rows in segments if agent_id == 2) == [[2.0], [22.0]]
    assert learner.replay.last_plan_metrics["cut_gap_count"] == 1


def test_fifo_overwrite_unlinks_old_generation_and_keeps_every_resident_row_once():
    replay = SequenceReplay(3, (1,), num_agents=1)
    for value in range(5):
        replay.add_sequence(
            torch.tensor([0]),
            torch.tensor([[float(value)]]),
            torch.tensor([value % 2]),
            torch.tensor([float(value)]),
            torch.tensor([[float(value + 1)]]),
            torch.tensor([0.9]),
            torch.tensor([False]),
            torch.tensor([False]),
        )

    segments = _planned_segments(replay, horizon=8, logical_batch_size=3)

    assert segments == [(0, [2.0, 3.0, 4.0])]
    metadata = replay.get_sequence_metadata(torch.tensor([0, 1, 2]))
    assert sorted(metadata["generations"].tolist()) == [1, 2, 2]
    assert replay.last_plan_metrics["cut_overwrite_count"] == 2
