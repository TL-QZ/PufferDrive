from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
import time

import numpy as np
import torch

from pufferlib import pufferl_sac
from pufferlib.sac_replay import TensorReplay


class _Profile:
    def __call__(self, *args, **kwargs):
        return None

    def end(self):
        return None


class _VecEnv:
    def __init__(self, observations, env_ids):
        self.observations = np.asarray(observations, dtype=np.float32)
        self.env_ids = np.asarray(env_ids, dtype=np.int64)
        self.received = 0
        self.sent_actions = []

    def recv(self):
        self.received += 1
        return (
            self.observations,
            np.arange(self.env_ids.size, dtype=np.float32) + 0.25,
            np.zeros(self.env_ids.size, dtype=np.bool_),
            np.zeros(self.env_ids.size, dtype=np.bool_),
            [],
            self.env_ids,
            np.ones(self.env_ids.size, dtype=np.bool_),
        )

    def send(self, actions):
        self.sent_actions.append(np.asarray(actions).copy())


class _Policy:
    def __init__(self):
        self.calls = 0

    def __call__(self, observations):
        self.calls += 1
        return torch.zeros((observations.shape[0], 3), device=observations.device)


class _Provider:
    def __init__(self, sources=None, boundary_step=None):
        self.sources = sources
        self.boundary_step = boundary_step
        self.calls = []

    def begin_collection(self, learner):
        self.calls.append(("begin", learner.agent_steps))
        return self.boundary_step

    def sample_actions(self, learner, observations):
        self.calls.append(("sample", observations.shape[0]))
        return torch.full((observations.shape[0],), 2), None

    def transition_sources(self, env_ids, masks):
        completed = self.learner.pending[env_ids] & masks
        self.calls.append(("transition_sources", completed.clone()))
        return self.sources

    def record_replay_sources(self, sources, indices):
        self.calls.append(("replay_sources", torch.as_tensor(sources).clone(), indices.clone()))

    def record_action_sources(self, env_ids):
        self.calls.append(("action_sources", env_ids.clone()))

    def end_collection(self, learner):
        self.calls.append(("end", learner.agent_steps))

    def metrics(self):
        return {"behavior/provider_calls": float(len(self.calls))}


def _make_learner(vecenv, *, batch_size, total_timesteps, world_size=1, provider=None):
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.profile = _Profile()
    learner.epoch = 0
    learner.replay = TensorReplay(batch_size, (2,))
    learner.batch_size = batch_size
    learner.refresh_count = batch_size
    learner.replay_eviction_order = None
    learner.replay_eviction_cursor = 0
    learner.config = {"total_timesteps": total_timesteps, "gamma": 0.9}
    learner.agent_steps = 0
    learner.global_step = 0
    learner.world_size = world_size
    learner.distributed = False
    learner.valid_inserts = 0
    learner.device = torch.device("cpu")
    learner.vecenv = vecenv
    learner.pending = torch.zeros(int(vecenv.env_ids.max()) + 1, dtype=torch.bool)
    learner.observations = torch.zeros((learner.pending.numel(), 2), dtype=torch.float32)
    learner.actions = torch.arange(learner.pending.numel(), dtype=torch.int64) % 3
    learner.stats = defaultdict(list)
    learner.env_continuous = False
    learner.policy = _Policy()
    learner.uncompiled_policy = learner.policy
    learner.amp_context = nullcontext()
    learner.pytorch = SimpleNamespace(
        sample_logits=lambda logits, **kwargs: (
            torch.ones(logits.shape[0], dtype=torch.long), None, None, None,
        )
    )
    learner.helpers = SimpleNamespace(
        pufferlib=SimpleNamespace(
            unroll_nested_dict=lambda info: [],
            utils=SimpleNamespace(reduce_environment_metrics=lambda stats: {}),
        )
    )
    if provider is not None:
        provider.learner = learner
        learner.behavior_policy = provider
    return learner


def test_default_collection_keeps_native_policy_sampling():
    vecenv = _VecEnv([[1, 2]], [0])
    learner = _make_learner(vecenv, batch_size=1, total_timesteps=1)

    learner.evaluate()

    assert learner.policy.calls == 1
    assert learner.actions.tolist() == [1]
    assert vecenv.sent_actions[-1].tolist() == [1]
    assert len(learner.replay) == 0


def test_provider_actions_and_quota_selected_sources_follow_replay_rows(monkeypatch):
    vecenv = _VecEnv([[100, 0], [101, 0], [102, 0], [103, 0]], [0, 1, 2, 3])
    provider = _Provider(sources=torch.tensor([10, 11, 12, 13]))
    learner = _make_learner(vecenv, batch_size=2, total_timesteps=4, provider=provider)
    learner.pending[:] = True
    learner.observations[:] = torch.tensor([[0, -1], [1, -1], [2, -1], [3, -1]])
    learner.actions[:] = torch.tensor([0, 1, 2, 0])
    monkeypatch.setattr(
        torch, "randperm", lambda size, **kwargs: torch.tensor([3, 1, 0, 2])[:size],
    )

    learner.evaluate()

    assert learner.policy.calls == 0
    assert learner.actions.tolist() == [2, 2, 2, 2]
    assert vecenv.sent_actions[-1].tolist() == [2, 2, 2, 2]
    torch.testing.assert_close(
        learner.replay.observations, torch.tensor([[3, -1], [1, -1]], dtype=torch.float32),
    )
    torch.testing.assert_close(learner.replay.actions, torch.tensor([0, 1]))
    source_event = next(call for call in provider.calls if call[0] == "replay_sources")
    assert source_event[1].tolist() == [13, 11]
    assert source_event[2].tolist() == [0, 1]
    assert provider.calls[-1][0] == "end"


def test_provider_sources_use_exact_refresh_destination_indices(monkeypatch):
    vecenv = _VecEnv([[200, 0], [201, 0]], [0, 1])
    provider = _Provider(sources=torch.tensor([20, 21]))
    learner = _make_learner(vecenv, batch_size=2, total_timesteps=2, provider=provider)
    learner.pending[:] = True
    learner.observations[:] = torch.tensor([[0, -1], [1, -1]])
    learner.actions[:] = torch.tensor([0, 1])
    learner.replay.add(
        torch.tensor([[-2, 0], [-1, 0]], dtype=torch.float32),
        torch.tensor([2, 2]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([[-2, 0], [-1, 0]], dtype=torch.float32),
        torch.tensor([0.9, 0.9]),
    )
    monkeypatch.setattr(torch, "randperm", lambda size, **kwargs: torch.tensor([1, 0])[:size])

    learner.evaluate()

    source_event = next(call for call in provider.calls if call[0] == "replay_sources")
    assert source_event[1].tolist() == [20, 21]
    assert source_event[2].tolist() == [1, 0]
    torch.testing.assert_close(
        learner.replay.observations, torch.tensor([[1, -1], [0, -1]], dtype=torch.float32),
    )


def test_provider_boundary_caps_received_budget_using_global_steps():
    vecenv = _VecEnv([[1, 2]], [0])
    provider = _Provider(boundary_step=3)
    learner = _make_learner(vecenv, batch_size=4, total_timesteps=20, world_size=2, provider=provider)

    learner.evaluate()

    assert vecenv.received == 2  # ceil((3 - 0) / 2)
    assert learner.agent_steps == 2
    assert provider.calls[0] == ("begin", 0)
    assert provider.calls[-1] == ("end", 2)


def test_provider_metrics_are_added_to_native_logger_payload():
    logged = []
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.distributed = False
    learner.stats = defaultdict(list)
    learner.last_stats = {}
    learner.last_stats_epoch = None
    learner.epoch = 1
    learner.losses = {}
    learner.profile = []
    learner.agent_steps = 4
    learner.global_step = 4
    learner.last_log_step = 4
    learner.last_log_time = time.time()
    learner.start_time = time.time()
    learner.device = torch.device("cpu")
    learner.optimizer = SimpleNamespace(param_groups=[{"lr": 0.1}])
    learner.alpha_optimizer = SimpleNamespace(param_groups=[{"lr": 0.002}])
    learner.replay = TensorReplay(2, (2,))
    learner.valid_inserts = 0
    learner.sampled_transitions = 0
    learner.learner_updates = 0
    learner.helpers = SimpleNamespace(
        dist_sum=lambda value, device: value,
        pufferlib=SimpleNamespace(utils=SimpleNamespace(reduce_environment_metrics=lambda stats: {})),
    )
    learner.rank = 0
    learner.logger = SimpleNamespace(log=lambda logs, step: logged.append((logs, step)))
    learner.behavior_policy = _Provider()

    logs = learner.mean_and_log()
    assert logs["alpha_learning_rate"] == 0.002

    assert logs["behavior/provider_calls"] == 0.0
    assert logged[0][0]["behavior/provider_calls"] == 0.0
    assert logged[0][1] == 4


def test_train_creates_provider_before_restoring_learner_state(tmp_path, monkeypatch):
    events = []
    learners = []
    args = {
        "data_dir": str(tmp_path),
        "device": "cpu",
        "total_timesteps": 0,
        "resume_state_path": "learner_state.pt",
    }

    class _Logger:
        def close(self, *_args):
            events.append("logger_close")

    class _Learner:
        def __init__(self, config, _vecenv, _policy, _critic, _replay, logger):
            self.config = {**config, "resume_state_path": args["resume_state_path"]}
            self.agent_steps = 0
            self.logger = logger
            self.closed = False
            learners.append(self)
            events.append("learner")

        def load_training_state(self, _path, same_run=False):
            events.append("restore")

        def close(self):
            self.closed = True
            events.append("close")
            return None

    monkeypatch.setattr(pufferl_sac, "_initialize_process_group", lambda *args: (False, "cpu", 1, 0))
    monkeypatch.setattr(pufferl_sac, "PuffeSAC", _Learner)

    provider = object()

    def make_provider(learner, startup_args):
        assert events == ["learner"]
        assert startup_args is args
        events.append("provider")
        return provider

    pufferl_sac.train(args=args, logger=_Logger(), behavior_policy_factory=make_provider)

    assert events == ["learner", "provider", "restore", "close", "logger_close"]
    assert learners[0].behavior_policy is provider

