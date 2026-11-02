"""Collection integration checks for discrete SAC n-step replay rows."""

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pufferlib import pufferl, pufferl_sac
from pufferlib.pufferl_sac_ppo_init_behavior_policy import validate_config as validate_behavior_config
from pufferlib.pufferl_sac_ppo_warm_start import validate_config as validate_warm_start_config
from pufferlib.sac_replay import NStepAccumulator, TensorReplay


class _Profile:
    def __call__(self, *args, **kwargs):
        return None

    def end(self):
        return None


class _SequenceVecEnv:
    single_observation_space = SimpleNamespace(shape=(1,))

    def __init__(
        self, *, masks=None, terminal_calls=(), truncation_calls=(), final_observation=99.0,
        provide_final_observation=True,
    ):
        self.masks = masks or {}
        self.terminal_calls = set(terminal_calls)
        self.truncation_calls = set(truncation_calls)
        self.final_observation = float(final_observation)
        self.provide_final_observation = provide_final_observation
        self.receive_count = 0
        self.sent_actions = []

    def recv(self):
        self.receive_count += 1
        call = self.receive_count
        infos = []
        truncation = call in self.truncation_calls
        if truncation and self.provide_final_observation:
            infos = [{pufferl_sac.FINAL_OBSERVATIONS_INFO_KEY: {
                "env_ids": np.array([0], dtype=np.int64),
                "observations": np.array([[self.final_observation]], dtype=np.float32),
            }}]
        return (
            np.array([[float(call)]], dtype=np.float32),
            np.array([float(call)], dtype=np.float32),
            np.array([call in self.terminal_calls], dtype=np.bool_),
            np.array([truncation], dtype=np.bool_),
            infos,
            np.array([0], dtype=np.int64),
            np.array([self.masks.get(call, True)], dtype=np.bool_),
        )

    def send(self, actions):
        self.sent_actions.append(np.asarray(actions).copy())


def _make_evaluate_learner(vecenv, *, n_steps, batch_size, total_timesteps, gamma=0.9):
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.profile = _Profile()
    learner.epoch = 0
    learner.replay = TensorReplay(batch_size, (1,))
    learner.batch_size = batch_size
    learner.refresh_count = batch_size
    learner.replay_eviction_order = None
    learner.replay_eviction_cursor = 0
    learner.config = {
        "total_timesteps": total_timesteps,
        "gamma": gamma,
        "n_steps": n_steps,
    }
    num_agents = getattr(vecenv, "num_agents", 1)
    learner.n_step_accumulator = (
        NStepAccumulator(num_agents, (1,), n_steps, gamma) if n_steps > 1 else None
    )
    learner.nstep_logged_horizon_sum = 0
    learner.nstep_logged_insert_count = 0
    learner.nstep_logged_quota_discarded = 0
    learner.nstep_logged_pending_starts = 0
    learner.global_step = 0
    learner.agent_steps = 0
    learner.world_size = 1
    learner.rank = 0
    learner.distributed = False
    learner.valid_inserts = 0
    learner.device = torch.device("cpu")
    learner.vecenv = vecenv
    learner.pending = torch.zeros(num_agents, dtype=torch.bool)
    learner.observations = torch.zeros((num_agents, 1), dtype=torch.float32)
    learner.actions = torch.zeros(num_agents, dtype=torch.int64)
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


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, observations):
        return self.linear(observations)


class _TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, observations):
        values = self.linear(observations)
        return values, values


class _ConstructorVecEnv:
    num_agents = 1
    agents_per_batch = 1
    single_observation_space = SimpleNamespace(shape=(1,))
    single_action_space = SimpleNamespace(n=2)


class _NoopUtilization:
    def stop(self):
        pass

    def join(self):
        pass


def _make_initialized_learner(monkeypatch, tmp_path, *, n_steps=1, replay=None, **overrides):
    monkeypatch.setattr(pufferl, "Utilization", _NoopUtilization)
    monkeypatch.setattr(pufferl.PuffeRL, "print_dashboard", lambda *_args, **_kwargs: None)
    config = {
        "data_dir": str(tmp_path),
        "action_count": 2,
        "total_timesteps": 20,
        "batch_size": 2,
        "minibatch_size": 2,
        "max_minibatch_size": 2,
        "checkpoint_interval": None,
        "n_steps": n_steps,
        **overrides,
    }
    replay = replay or TensorReplay(config["batch_size"], (1,))
    return pufferl_sac.PuffeSAC(
        config,
        _ConstructorVecEnv(),
        _TinyActor(),
        _TinyCritic(),
        replay,
        reset_env=False,
    )


def test_collection_accumulates_overlapping_windows_and_preserves_them_between_calls():
    vecenv = _SequenceVecEnv()
    learner = _make_evaluate_learner(vecenv, n_steps=3, batch_size=1, total_timesteps=20)

    learner.evaluate()

    assert vecenv.receive_count == 4
    assert len(learner.replay) == 1
    first = learner.replay.get_batch(torch.tensor([0]))
    torch.testing.assert_close(first[0], torch.tensor([[1.0]]))
    torch.testing.assert_close(first[2], torch.tensor([2.0 + 0.9 * 3.0 + 0.81 * 4.0]))
    torch.testing.assert_close(first[3], torch.tensor([[4.0]]))
    torch.testing.assert_close(first[4], torch.tensor([0.9 ** 3]))
    assert learner.n_step_accumulator.pending_count == 2

    learner.evaluate()

    assert vecenv.receive_count == 5
    second = learner.replay.get_batch(torch.tensor([0]))
    torch.testing.assert_close(second[0], torch.tensor([[2.0]]))
    torch.testing.assert_close(second[2], torch.tensor([3.0 + 0.9 * 4.0 + 0.81 * 5.0]))
    torch.testing.assert_close(second[3], torch.tensor([[5.0]]))
    torch.testing.assert_close(second[4], torch.tensor([0.9 ** 3]))
    assert learner.valid_inserts == 2


def test_invalid_mask_flushes_older_prefix_at_last_valid_successor():
    vecenv = _SequenceVecEnv(
        masks={3: False}, truncation_calls={3}, provide_final_observation=False,
    )
    learner = _make_evaluate_learner(vecenv, n_steps=3, batch_size=8, total_timesteps=3)

    learner.evaluate()

    assert vecenv.receive_count == 3
    assert len(learner.replay) == 1
    row = learner.replay.get_batch(torch.tensor([0]))
    torch.testing.assert_close(row[0], torch.tensor([[1.0]]))
    torch.testing.assert_close(row[2], torch.tensor([2.0]))
    torch.testing.assert_close(row[3], torch.tensor([[2.0]]))
    torch.testing.assert_close(row[4], torch.tensor([0.9]))
    assert learner.n_step_accumulator.pending_count == 0


def test_compact_rows_keep_shuffled_noncontiguous_agent_ids_aligned_across_mask_gap():
    class _InterleavedVecEnv:
        num_agents = 5
        single_observation_space = SimpleNamespace(shape=(1,))

        def __init__(self):
            self.receive_count = 0

        def recv(self):
            self.receive_count += 1
            if self.receive_count == 1:
                agent_ids = [4, 1]
                observations = [[40.0], [10.0]]
                rewards = [0.0, 0.0]
                masks = [True, True]
            elif self.receive_count == 2:
                agent_ids = [1, 4]
                observations = [[11.0], [41.0]]
                rewards = [1.0, 10.0]
                masks = [True, True]
            else:
                agent_ids = [4, 1]
                observations = [[42.0], [12.0]]
                rewards = [20.0, 2.0]
                masks = [False, True]
            row_count = len(agent_ids)
            return (
                np.asarray(observations, dtype=np.float32),
                np.asarray(rewards, dtype=np.float32),
                np.zeros(row_count, dtype=np.bool_),
                np.zeros(row_count, dtype=np.bool_),
                [],
                np.asarray(agent_ids, dtype=np.int64),
                np.asarray(masks, dtype=np.bool_),
            )

        def send(self, actions):
            return None

    vecenv = _InterleavedVecEnv()
    learner = _make_evaluate_learner(vecenv, n_steps=2, batch_size=2, total_timesteps=20)

    learner.evaluate()

    assert vecenv.receive_count == 3
    assert len(learner.replay) == 2
    rows = learner.replay.get_batch(torch.arange(2))
    torch.testing.assert_close(rows[0], torch.tensor([[40.0], [10.0]]))
    torch.testing.assert_close(rows[2], torch.tensor([10.0, 1.0 + 0.9 * 2.0]))
    torch.testing.assert_close(rows[3], torch.tensor([[41.0], [12.0]]))
    torch.testing.assert_close(rows[4], torch.tensor([0.9, 0.9 ** 2]))


def test_true_terminal_wins_over_simultaneous_truncation_without_final_info():
    vecenv = _SequenceVecEnv(terminal_calls={3}, truncation_calls={3}, provide_final_observation=False)
    learner = _make_evaluate_learner(vecenv, n_steps=3, batch_size=8, total_timesteps=3)

    learner.evaluate()

    assert len(learner.replay) == 2
    rows = learner.replay.get_batch(torch.arange(2))
    torch.testing.assert_close(rows[0], torch.tensor([[2.0], [1.0]]))
    torch.testing.assert_close(rows[2], torch.tensor([3.0, 2.0 + 0.9 * 3.0]))
    torch.testing.assert_close(rows[4], torch.zeros(2))


def test_truncation_flushes_each_prefix_to_pre_reset_final_observation():
    vecenv = _SequenceVecEnv(truncation_calls={3}, final_observation=99.0)
    learner = _make_evaluate_learner(vecenv, n_steps=3, batch_size=8, total_timesteps=3)

    learner.evaluate()

    assert vecenv.receive_count == 3
    assert len(learner.replay) == 2
    rows = learner.replay.get_batch(torch.arange(2))
    torch.testing.assert_close(rows[0], torch.tensor([[2.0], [1.0]]))
    torch.testing.assert_close(rows[2], torch.tensor([3.0, 2.0 + 0.9 * 3.0]))
    torch.testing.assert_close(rows[3], torch.tensor([[99.0], [99.0]]))
    torch.testing.assert_close(rows[4], torch.tensor([0.9, 0.9 ** 2]))


def test_truncation_without_final_observation_still_fails_for_valid_row():
    vecenv = _SequenceVecEnv(truncation_calls={3}, provide_final_observation=False)
    learner = _make_evaluate_learner(vecenv, n_steps=3, batch_size=8, total_timesteps=3)

    with pytest.raises(ValueError, match="Missing pre-reset final observations"):
        learner.evaluate()


def test_terminal_tail_quota_discards_excess_candidates_without_reemitting(monkeypatch):
    vecenv = _SequenceVecEnv(terminal_calls={4})
    learner = _make_evaluate_learner(vecenv, n_steps=3, batch_size=1, total_timesteps=5)
    monkeypatch.setattr(torch, "randperm", lambda size, **kwargs: torch.arange(size))

    learner.evaluate()

    assert vecenv.receive_count == 4
    assert len(learner.replay) == 1
    first = learner.replay.get_batch(torch.tensor([0]))
    torch.testing.assert_close(first[0], torch.tensor([[3.0]]))
    torch.testing.assert_close(first[2], torch.tensor([4.0]))
    torch.testing.assert_close(first[4], torch.tensor([0.0]))
    assert learner.nstep_logged_quota_discarded == 2
    assert learner.n_step_accumulator.pending_count == 0

    learner.evaluate()

    assert vecenv.receive_count == 5
    next_episode = learner.replay.get_batch(torch.tensor([0]))
    torch.testing.assert_close(next_episode[0], torch.tensor([[4.0]]))
    torch.testing.assert_close(next_episode[2], torch.tensor([5.0]))
    torch.testing.assert_close(next_episode[4], torch.tensor([0.9]))
    assert learner.valid_inserts == 2


def test_global_received_budget_flushes_short_windows_without_an_extra_receive():
    vecenv = _SequenceVecEnv()
    learner = _make_evaluate_learner(vecenv, n_steps=4, batch_size=8, total_timesteps=3)

    learner.evaluate()

    assert vecenv.receive_count == 3
    assert learner.agent_steps == 3
    assert len(learner.replay) == 2
    rows = learner.replay.get_batch(torch.arange(2))
    torch.testing.assert_close(rows[0], torch.tensor([[2.0], [1.0]]))
    torch.testing.assert_close(rows[2], torch.tensor([3.0, 2.0 + 0.9 * 3.0]))
    torch.testing.assert_close(rows[3], torch.tensor([[3.0], [3.0]]))
    torch.testing.assert_close(rows[4], torch.tensor([0.9, 0.9 ** 2]))
    assert learner.n_step_accumulator.pending_count == 0


@pytest.mark.parametrize("critic_entropy_in_target", [True, False])
def test_n_step_replay_row_uses_aggregated_reward_and_endpoint_soft_bootstrap(
    critic_entropy_in_target,
):
    accumulator = NStepAccumulator(1, (1,), n_steps=3, gamma=0.9)
    accumulator.append(
        torch.tensor([0]), torch.tensor([[1.0]]), torch.tensor([1]),
        torch.tensor([2.0]), torch.tensor([[2.0]]),
        torch.tensor([False]), torch.tensor([False]),
    )
    rows, horizons = accumulator.append(
        torch.tensor([0]), torch.tensor([[2.0]]), torch.tensor([0]),
        torch.tensor([3.0]), torch.tensor([[99.0]]),
        torch.tensor([False]), torch.tensor([True]),
    )
    # Flush output is grouped by effective horizon; index one is the 2-step row.
    assert horizons.tolist() == [1, 2]
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.amp_context = nullcontext()
    learner.config = {"critic_entropy_in_target": critic_entropy_in_target}
    learner.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
    learner.policy = lambda observations: torch.zeros((observations.shape[0], 2))
    learner.target_critic = lambda observations: (
        torch.tensor([[1.0, 3.0]]).expand(observations.shape[0], -1),
        torch.tensor([[1.0, 3.0]]).expand(observations.shape[0], -1),
    )

    target = learner._compute_targets(
        rows[2][1:2], rows[3][1:2], rows[4][1:2],
    )
    soft_value = 2.0 + (np.log(2.0) if critic_entropy_in_target else 0.0)
    expected = 2.0 + 0.9 * 3.0 + (0.9 ** 2) * soft_value
    torch.testing.assert_close(target, torch.tensor([expected], dtype=torch.float32))


@pytest.mark.parametrize("n_steps", [True, 0, -1, 1.5, "3"])
def test_sac_rejects_invalid_n_steps(monkeypatch, tmp_path, n_steps):
    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        _make_initialized_learner(monkeypatch, tmp_path, n_steps=n_steps)


def test_n_step_mode_requires_empty_initial_replay(monkeypatch, tmp_path):
    replay = TensorReplay(2, (1,))
    replay.add(
        torch.tensor([[0.0]]), torch.tensor([0]), torch.tensor([1.0]),
        torch.tensor([[1.0]]), torch.tensor([0.9]),
    )
    with pytest.raises(ValueError, match="empty initial replay"):
        _make_initialized_learner(monkeypatch, tmp_path, n_steps=3, replay=replay)


def test_n_step_one_keeps_the_existing_one_step_collection_tuple():
    vecenv = _SequenceVecEnv()
    learner = _make_evaluate_learner(vecenv, n_steps=1, batch_size=1, total_timesteps=2)
    assert learner.n_step_accumulator is None
    torch.manual_seed(41)
    expected_after = torch.rand(1)
    torch.manual_seed(41)

    learner.evaluate()
    learner.evaluate()

    actual_after = torch.rand(1)
    torch.testing.assert_close(actual_after, expected_after)
    row = learner.replay.get_batch(torch.tensor([0]))
    torch.testing.assert_close(row[0], torch.tensor([[1.0]]))
    torch.testing.assert_close(row[2], torch.tensor([2.0]))
    torch.testing.assert_close(row[3], torch.tensor([[2.0]]))
    torch.testing.assert_close(row[4], torch.tensor([0.9]))


def test_one_step_refresh_uses_only_the_existing_eviction_permutation_rng(monkeypatch):
    vecenv = _SequenceVecEnv()
    learner = _make_evaluate_learner(vecenv, n_steps=1, batch_size=2, total_timesteps=1)
    learner.refresh_count = 1
    learner.pending[0] = True
    learner.observations[0] = torch.tensor([7.0])
    learner.replay.add(
        torch.tensor([[-1.0], [-2.0]]), torch.tensor([0, 1]), torch.tensor([-1.0, -2.0]),
        torch.tensor([[-1.0], [-2.0]]), torch.tensor([0.9, 0.9]),
    )
    original_randperm = torch.randperm
    calls = []

    def track_randperm(size, **kwargs):
        calls.append(size)
        return original_randperm(size, **kwargs)

    monkeypatch.setattr(torch, "randperm", track_randperm)
    torch.manual_seed(53)
    expected_order = torch.randperm(2)
    expected_after = torch.rand(1)
    torch.manual_seed(53)
    calls.clear()

    learner.evaluate()

    actual_after = torch.rand(1)
    torch.testing.assert_close(actual_after, expected_after)
    assert calls == [2]
    assert learner.replay_eviction_order.tolist() == expected_order.tolist()
    assert learner.replay_eviction_cursor == 1
    updated_slot = expected_order[0].item()
    row = learner.replay.get_batch(torch.tensor([updated_slot]))
    torch.testing.assert_close(row[0], torch.tensor([[7.0]]))
    torch.testing.assert_close(row[2], torch.tensor([1.0]))
    torch.testing.assert_close(row[3], torch.tensor([[1.0]]))
    torch.testing.assert_close(row[4], torch.tensor([0.9]))


def test_directly_attached_behavior_provider_is_rejected_before_collection_hooks():
    class _Provider:
        called = False

        def begin_collection(self, learner):
            self.called = True

    learner = _make_evaluate_learner(
        _SequenceVecEnv(), n_steps=3, batch_size=1, total_timesteps=3,
    )
    provider = _Provider()
    learner.behavior_policy = provider

    with pytest.raises(ValueError, match="PPO behavior-policy providers"):
        learner.evaluate()

    assert not provider.called


def test_train_preflight_rejects_behavior_factory_and_invalid_horizon_before_env_creation(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(
        pufferl_sac, "_initialize_process_group",
        lambda *_args, **_kwargs: pytest.fail("process group started before n_steps preflight"),
    )
    with pytest.raises(ValueError, match="PPO behavior_policy_factory"):
        pufferl_sac.train(
            args={"train": {"n_steps": 3}},
            behavior_policy_factory=lambda *_args: pytest.fail("provider factory was called"),
        )

    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        pufferl_sac.train(args={"train": {"n_steps": 0}})


def test_both_ppo_collection_preflights_reject_n_step_mode(tmp_path):
    with pytest.raises(ValueError, match="train.n_steps=1"):
        validate_behavior_config({"train": {"n_steps": 3}})

    config = _valid_warm_start_config(tmp_path)
    with pytest.raises(ValueError, match="sac.train.n_steps=1"):
        validate_warm_start_config(config, world_size=1)


def test_n_step_logging_averages_inserted_horizons_and_keeps_pending_gauge(monkeypatch, tmp_path):
    learner = _make_initialized_learner(monkeypatch, tmp_path, n_steps=3)
    learner.nstep_logged_horizon_sum = 7
    learner.nstep_logged_insert_count = 2
    learner.nstep_logged_quota_discarded = 4
    learner.nstep_logged_pending_starts = 5

    logs = learner.mean_and_log()

    assert logs["sac/n_steps"] == 3
    assert logs["replay/mean_effective_horizon"] == pytest.approx(3.5)
    assert logs["replay/quota_discarded"] == 4
    assert logs["replay/pending_starts"] == 5
    assert learner.nstep_logged_horizon_sum == 0
    assert learner.nstep_logged_insert_count == 0
    assert learner.nstep_logged_quota_discarded == 0

    next_logs = learner.mean_and_log()
    assert next_logs["replay/mean_effective_horizon"] == 0
    assert next_logs["replay/quota_discarded"] == 0
    assert next_logs["replay/pending_starts"] == 5


def test_legacy_checkpoint_defaults_to_one_step_and_n_step_resume_rejects_horizon_change(
    monkeypatch, tmp_path,
):
    legacy_source = _make_initialized_learner(
        monkeypatch, tmp_path / "legacy-source", n_steps=1,
    )
    legacy_source.save_checkpoint()
    legacy_path = tmp_path / "legacy-source" / "trainer_state.pt"
    legacy_state = torch.load(legacy_path, map_location="cpu", weights_only=False)
    legacy_state["config"].pop("n_steps")
    torch.save(legacy_state, legacy_path)

    legacy_restored = _make_initialized_learner(
        monkeypatch, tmp_path / "legacy-restored", n_steps=1,
    )
    legacy_restored.load_training_state(str(legacy_path))
    assert legacy_restored.config["n_steps"] == 1
    assert legacy_restored.n_step_accumulator is None
    assert len(legacy_restored.replay) == 0

    n_step_source = _make_initialized_learner(
        monkeypatch, tmp_path / "nstep-source", n_steps=3,
    )
    n_step_source.n_step_accumulator.append(
        torch.tensor([0]), torch.tensor([[1.0]]), torch.tensor([0]),
        torch.tensor([2.0]), torch.tensor([[2.0]]),
        torch.tensor([False]), torch.tensor([False]),
    )
    n_step_source.save_checkpoint()
    n_step_path = tmp_path / "nstep-source" / "trainer_state.pt"
    assert "n_step_accumulator" not in torch.load(
        n_step_path, map_location="cpu", weights_only=False,
    )
    n_step_restored = _make_initialized_learner(
        monkeypatch, tmp_path / "nstep-restored", n_steps=3,
    )
    n_step_restored.load_training_state(str(n_step_path))
    assert n_step_restored.config["n_steps"] == 3
    assert n_step_restored.n_step_accumulator.pending_count == 0
    assert not n_step_restored.pending.any()
    assert len(n_step_restored.replay) == 0

    mismatched = _make_initialized_learner(
        monkeypatch, tmp_path / "nstep-restored", n_steps=4,
    )
    with pytest.raises(ValueError, match="Checkpoint config mismatch: n_steps"):
        mismatched.load_training_state(str(n_step_path))


def _valid_warm_start_config(tmp_path):
    shared_env = {"capture_final_observations": True}
    shared_policy = {"action_type": "discrete"}
    return {
        "ppo": {
            "train": {"total_timesteps": 10},
            "policy": dict(shared_policy),
            "env": dict(shared_env),
        },
        "sac": {
            "train": {
                "total_timesteps": 15,
                "learn_alpha": False,
                "initial_alpha": 0.1,
                "anneal_lr": False,
                "n_steps": 3,
            },
            "policy_name": "DriveSAC",
            "policy": dict(shared_policy),
            "env": dict(shared_env),
        },
        "warm_start": {"handoff_steps": 5, "sac_steps": 10, "run_dir": str(tmp_path)},
        "provenance": {"expected_world_size": 1},
    }


def test_ppo_warm_start_preflight_rejects_n_step_sac(tmp_path):
    config = _valid_warm_start_config(tmp_path)
    with pytest.raises(ValueError, match="n_steps=1"):
        validate_warm_start_config(config, world_size=1)
