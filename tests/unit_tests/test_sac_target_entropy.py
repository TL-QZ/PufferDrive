"""Focused coverage for adaptive discrete-SAC target entropy."""

from collections import defaultdict
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pufferlib import pufferl, pufferl_sac
from pufferlib.sac_replay import TensorReplay


class _TinyActor(nn.Module):
    action_count = 2
    is_continuous = False

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
            self.linear.bias.zero_()

    def forward(self, observations):
        return self.linear(observations)


class _TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Linear(3, 2)
        self.q2 = nn.Linear(3, 2)

    def forward(self, observations):
        return self.q1(observations), self.q2(observations)


class _FakeVecEnv:
    def __init__(self, action_count=2):
        self.agents_per_batch = 4
        self.num_agents = 4
        self.single_observation_space = SimpleNamespace(shape=(3,))
        self.single_action_space = SimpleNamespace(n=action_count)
        self.closed = False

    def async_reset(self, seed):
        self.seed = seed

    def close(self):
        self.closed = True


class _DummyUtilization:
    def stop(self):
        pass

    def join(self):
        pass


def _patch_learner_helpers(monkeypatch):
    monkeypatch.setattr(pufferl, "Utilization", _DummyUtilization)
    monkeypatch.setattr(pufferl.PuffeRL, "print_dashboard", lambda *_args, **_kwargs: None)


def _make_learner(monkeypatch, data_dir, **config_updates):
    _patch_learner_helpers(monkeypatch)
    replay = config_updates.pop("replay", None)
    config = {
        "data_dir": str(data_dir),
        "action_count": 2,
        "total_timesteps": 32,
        "batch_size": 8,
        "minibatch_size": 8,
        "max_minibatch_size": 8,
        "checkpoint_interval": None,
        **config_updates,
    }
    return pufferl_sac.PuffeSAC(
        config,
        _FakeVecEnv(config["action_count"]),
        _TinyActor(),
        _TinyCritic(),
        replay if replay is not None else TensorReplay(config["batch_size"], (3,)),
    )


def _fill_replay(learner):
    observations = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
    learner.replay.add(
        observations,
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        torch.tensor([0.2, -0.1, 0.5, 0.3, -0.2, 0.4, 0.1, 0.0]),
        observations * 0.7 + 0.15,
        torch.zeros(8),
    )


def test_wandb_receives_current_target_entropy_after_reduction(monkeypatch, tmp_path):
    learner = _make_learner(
        monkeypatch, tmp_path,
        target_entropy=0.5,
        target_entropy_schedule="adaptive",
        target_entropy_mean_threshold=100.0,
        target_entropy_std_threshold=100.0,
        target_entropy_decay_factor=0.5,
        target_entropy_required_count=1,
    )
    logged_records = []
    wandb_logger = object.__new__(pufferl.WandbLogger)
    wandb_logger.wandb = SimpleNamespace(
        log=lambda logs, *, step: logged_records.append((dict(logs), step)),
    )
    original_logger = learner.logger
    learner.logger = wandb_logger
    try:
        learner.mean_and_log()
        _fill_replay(learner)
        learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
        learner.agent_steps = 32
        learner.mean_and_log()

        initial_logs, initial_step = logged_records[0]
        updated_logs, updated_step = logged_records[1]
        assert initial_logs["losses/target_entropy"] == pytest.approx(0.5)
        assert updated_logs["losses/target_entropy"] == pytest.approx(0.25)
        assert updated_logs["target_entropy/decay_count"] == 1
        assert (initial_step, updated_step) == (0, 32)
        assert learner.config["target_entropy"] == 0.5
    finally:
        learner.logger = original_logger
        learner.close()


def _controller(**overrides):
    parameters = {
        "ema_decay": 0.5,
        "mean_threshold": 1e-12,
        "std_threshold": 1e-12,
        "decay_factor": 0.5,
        "required_count": 100,
    }
    parameters.update(overrides)
    force_decay_steps = parameters.pop("force_decay_steps", None)
    return pufferl_sac.AdaptiveTargetEntropyController(
        0.5,
        force_decay_steps=force_decay_steps,
        **parameters,
    )


def test_controller_uses_the_specified_ema_mean_and_variance_recurrence():
    controller = _controller()
    expected = controller.state_dict()

    for entropy in (0.25, 0.75, 0.4):
        delta = entropy - expected["mean_entropy"]
        expected["mean_entropy"] += (1 - expected["ema_decay"]) * delta
        expected["variance"] = expected["ema_decay"] * (
            expected["variance"] + (1 - expected["ema_decay"]) * delta**2
        )
        controller.observe(entropy)
        actual = controller.state_dict()
        assert actual["mean_entropy"] == pytest.approx(expected["mean_entropy"])
        assert actual["variance"] == pytest.approx(expected["variance"])
        assert actual["target_entropy"] == pytest.approx(0.5)
        assert actual["qualifying_count"] == 0


def test_controller_uses_strict_mean_boundary_and_inclusive_std_boundary():
    mean_boundary = _controller(
        ema_decay=0.0,
        mean_threshold=0.25,
        std_threshold=0.0,
        required_count=2,
    )
    mean_boundary.observe(0.75)
    assert mean_boundary.state_dict()["qualifying_count"] == 0

    std_boundary = _controller(
        ema_decay=0.5,
        mean_threshold=0.2,
        std_threshold=0.125,
        required_count=2,
    )
    std_boundary.observe(0.75)
    assert std_boundary.state_dict()["mean_entropy"] == pytest.approx(0.625)
    assert std_boundary.state_dict()["variance"] == pytest.approx(0.125**2)
    assert std_boundary.state_dict()["qualifying_count"] == 1


def test_controller_retains_qualifying_count_across_failed_observations():
    controller = _controller(
        ema_decay=0.0,
        mean_threshold=0.2,
        std_threshold=0.0,
        decay_factor=0.5,
        required_count=2,
    )

    controller.observe(0.5)
    assert controller.state_dict()["qualifying_count"] == 1
    controller.observe(0.75)
    assert controller.state_dict()["qualifying_count"] == 1
    assert controller.state_dict()["target_entropy"] == pytest.approx(0.5)
    controller.observe(0.5)
    state = controller.state_dict()
    assert state["target_entropy"] == pytest.approx(0.25)
    assert state["qualifying_count"] == 0
    assert state["decay_count"] == 1


def test_controller_state_dict_continues_after_restore():
    source = _controller(required_count=3)
    source.observe(0.5)
    source.observe(0.75)
    restored = _controller(required_count=3)

    restored.load_state_dict(source.state_dict())
    source.observe(0.5)
    restored.observe(0.5)

    assert restored.state_dict() == source.state_dict()


def test_forced_decay_uses_inclusive_agent_step_boundary_and_does_not_repeat():
    interval = 500_000_000
    controller = _controller(
        force_decay_steps=interval,
        mean_threshold=1e-12,
        std_threshold=0.0,
        required_count=100,
    )

    controller.observe(0.9, agent_steps=interval - 1)
    before_boundary = controller.state_dict()
    assert before_boundary["target_entropy"] == pytest.approx(0.5)
    assert before_boundary["forced_decay_count"] == 0

    controller.observe(0.9, agent_steps=interval)
    at_boundary = controller.state_dict()
    assert at_boundary["target_entropy"] == pytest.approx(0.25)
    assert at_boundary["forced_decay_count"] == 1
    assert at_boundary["last_decay_agent_steps"] == interval

    controller.observe(0.9, agent_steps=interval)
    repeated_boundary = controller.state_dict()
    assert repeated_boundary["target_entropy"] == pytest.approx(0.25)
    assert repeated_boundary["forced_decay_count"] == 1


def test_normal_decay_resets_the_forced_decay_interval():
    interval = 500_000_000
    normal_decay_step = 100_000_000
    controller = _controller(
        ema_decay=0.0,
        mean_threshold=0.1,
        std_threshold=0.0,
        decay_factor=0.5,
        required_count=1,
        force_decay_steps=interval,
    )

    controller.observe(0.45, agent_steps=normal_decay_step)
    after_normal_decay = controller.state_dict()
    assert after_normal_decay["target_entropy"] == pytest.approx(0.25)
    assert after_normal_decay["decay_count"] == 1
    assert after_normal_decay["forced_decay_count"] == 0
    assert after_normal_decay["last_decay_agent_steps"] == normal_decay_step

    controller.observe(0.45, agent_steps=normal_decay_step + interval - 1)
    before_forced_boundary = controller.state_dict()
    assert before_forced_boundary["target_entropy"] == pytest.approx(0.25)
    assert before_forced_boundary["forced_decay_count"] == 0

    controller.observe(0.45, agent_steps=normal_decay_step + interval)
    at_forced_boundary = controller.state_dict()
    assert at_forced_boundary["target_entropy"] == pytest.approx(0.125)
    assert at_forced_boundary["forced_decay_count"] == 1
    assert at_forced_boundary["last_decay_agent_steps"] == normal_decay_step + interval


def test_normal_and_forced_triggers_at_the_same_step_decay_only_once():
    controller = _controller(
        mean_threshold=100.0,
        std_threshold=100.0,
        required_count=1,
        force_decay_steps=500_000_000,
    )

    controller.observe(0.5, agent_steps=500_000_000)
    state = controller.state_dict()
    assert state["target_entropy"] == pytest.approx(0.25)
    assert state["decay_count"] == 1
    assert state["forced_decay_count"] == 0
    assert state["last_decay_agent_steps"] == 500_000_000


def test_large_agent_step_jump_causes_one_decay_without_catch_up():
    controller = _controller(
        mean_threshold=1e-12,
        std_threshold=0.0,
        required_count=100,
        force_decay_steps=500_000_000,
    )

    controller.observe(0.9, agent_steps=10_000_000_000)
    state = controller.state_dict()
    assert state["target_entropy"] == pytest.approx(0.25)
    assert state["forced_decay_count"] == 1
    assert state["last_decay_agent_steps"] == 10_000_000_000


def test_forced_decay_resets_qualifying_count_and_keeps_entropy_ema():
    controller = _controller(
        ema_decay=0.9,
        mean_threshold=0.1,
        std_threshold=10.0,
        required_count=100,
        force_decay_steps=10,
    )
    controller.observe(0.5, agent_steps=1)
    controller.observe(0.5, agent_steps=2)
    assert controller.state_dict()["qualifying_count"] == 2

    controller.observe(0.0, agent_steps=10)
    state = controller.state_dict()
    assert state["target_entropy"] == pytest.approx(0.25)
    assert state["forced_decay_count"] == 1
    assert state["qualifying_count"] == 0
    assert state["mean_entropy"] == pytest.approx(0.45)
    assert state["variance"] == pytest.approx(0.0225)


def test_hydra_sac_config_accepts_all_adaptive_entropy_overrides():
    args = pufferl_sac.load_config(overrides=[
        "train.target_entropy_schedule=adaptive",
        "train.target_entropy_ema_decay=0.9",
        "train.target_entropy_mean_threshold=0.03",
        "train.target_entropy_std_threshold=0.04",
        "train.target_entropy_decay_factor=0.8",
        "train.target_entropy_required_count=7",
        "train.target_entropy_force_decay_steps=500000000",
    ])

    assert args["train"]["target_entropy_schedule"] == "adaptive"
    assert args["train"]["target_entropy_ema_decay"] == pytest.approx(0.9)
    assert args["train"]["target_entropy_mean_threshold"] == pytest.approx(0.03)
    assert args["train"]["target_entropy_std_threshold"] == pytest.approx(0.04)
    assert args["train"]["target_entropy_decay_factor"] == pytest.approx(0.8)
    assert args["train"]["target_entropy_required_count"] == 7
    assert args["train"]["target_entropy_force_decay_steps"] == 500_000_000


@pytest.mark.parametrize(
    "config_updates, message",
    [
        ({"target_entropy_schedule": "unknown"}, "target_entropy_schedule"),
        ({"target_entropy_ema_decay": 1.0}, "target_entropy_ema_decay"),
        ({"target_entropy_mean_threshold": 0.0}, "target_entropy_mean_threshold"),
        ({"target_entropy_std_threshold": float("nan")}, "target_entropy_std_threshold"),
        ({"target_entropy_decay_factor": 1.0}, "target_entropy_decay_factor"),
        ({"target_entropy_required_count": 0}, "target_entropy_required_count"),
        ({"target_entropy_force_decay_steps": -1}, "target_entropy_force_decay_steps"),
        ({"target_entropy_force_decay_steps": True}, "target_entropy_force_decay_steps"),
        ({"target_entropy_force_decay_steps": 1.5}, "target_entropy_force_decay_steps"),
        ({"target_entropy_force_decay_steps": 0}, "target_entropy_force_decay_steps"),
        ({"target_entropy_schedule": "constant", "target_entropy_force_decay_steps": 10},
         "requires target_entropy_schedule='adaptive'"),
        ({"target_entropy_schedule": "adaptive", "learn_alpha": False}, "requires learn_alpha=True"),
    ],
)
def test_learner_rejects_invalid_adaptive_entropy_config(monkeypatch, tmp_path, config_updates, message):
    config = {"target_entropy_schedule": "adaptive", **config_updates}
    with pytest.raises(ValueError, match=message):
        _make_learner(monkeypatch, tmp_path, **config)


def test_learner_derives_automatic_required_count_from_local_update_geometry(monkeypatch, tmp_path):
    learner = _make_learner(
        monkeypatch,
        tmp_path,
        batch_size=7,
        minibatch_size=4,
        max_minibatch_size=3,
        update_epochs=3,
        target_entropy=0.5,
        target_entropy_schedule="adaptive",
        target_entropy_required_count=None,
    )
    try:
        assert learner.target_entropy_controller.state_dict()["required_count"] == 6
    finally:
        learner.close()


def test_adaptive_target_is_invariant_to_policy_microbatch_partition(monkeypatch, tmp_path):
    shared_config = {
        "target_entropy": 0.5,
        "target_entropy_schedule": "adaptive",
        "target_entropy_mean_threshold": 100.0,
        "target_entropy_std_threshold": 100.0,
        "target_entropy_required_count": 100,
        "update_epochs": 2,
    }
    full_batch = _make_learner(
        monkeypatch,
        tmp_path / "full-batch",
        max_minibatch_size=8,
        **shared_config,
    )
    microbatched = _make_learner(
        monkeypatch,
        tmp_path / "microbatched",
        max_minibatch_size=3,
        **shared_config,
    )
    _fill_replay(full_batch)
    _fill_replay(microbatched)

    try:
        torch.manual_seed(101)
        full_batch._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
        torch.manual_seed(101)
        microbatched._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)

        full_state = full_batch.target_entropy_controller.state_dict()
        micro_state = microbatched.target_entropy_controller.state_dict()
        for key in ("target_entropy", "mean_entropy", "variance"):
            assert micro_state[key] == pytest.approx(full_state[key], rel=1e-5, abs=1e-7)
        assert full_state["qualifying_count"] == micro_state["qualifying_count"] == 2
        assert full_state["decay_count"] == micro_state["decay_count"] == 0
    finally:
        full_batch.close()
        microbatched.close()


def test_null_target_entropy_keeps_the_existing_action_count_default(monkeypatch, tmp_path):
    learner = _make_learner(
        monkeypatch,
        tmp_path,
        action_count=3,
        target_entropy=None,
        target_entropy_schedule="adaptive",
    )
    try:
        assert learner.config["target_entropy"] == pytest.approx(0.98 * math.log(3))
        assert learner.target_entropy_controller.state_dict()["initial_target_entropy"] == pytest.approx(
            0.98 * math.log(3)
        )
    finally:
        learner.close()


def test_controller_observation_runs_after_alpha_update_and_only_with_policy_update(monkeypatch, tmp_path):
    learner = _make_learner(
        monkeypatch,
        tmp_path,
        target_entropy=0.5,
        target_entropy_schedule="adaptive",
        target_entropy_mean_threshold=100.0,
        target_entropy_std_threshold=100.0,
        target_entropy_decay_factor=0.5,
        target_entropy_required_count=1,
    )
    _fill_replay(learner)
    captured_gradients = []
    hook = learner.log_alpha.register_hook(
        lambda gradient: captured_gradients.append(gradient.detach().clone()) or gradient
    )
    try:
        first_target = learner.config["target_entropy"]
        first_losses = defaultdict(float)
        learner._train_sac_transition(first_losses, lambda *_args, **_kwargs: None, 0)
        assert captured_gradients[0].item() == pytest.approx(first_losses["entropy"] - first_target)
        assert learner.target_entropy_controller.state_dict()["target_entropy"] == pytest.approx(0.25)

        second_target = learner.target_entropy_controller.state_dict()["target_entropy"]
        second_losses = defaultdict(float)
        learner._train_sac_transition(second_losses, lambda *_args, **_kwargs: None, 1)
        assert captured_gradients[1].item() == pytest.approx(second_losses["entropy"] - second_target)

        state_before_critic_only_update = learner.target_entropy_controller.state_dict()
        learner._train_sac_transition(
            defaultdict(float), lambda *_args, **_kwargs: None, 2, update_policy=False,
        )
        assert learner.target_entropy_controller.state_dict() == state_before_critic_only_update
    finally:
        hook.remove()
        learner.close()


def test_learner_forces_decay_from_received_agent_steps_not_update_counters(monkeypatch, tmp_path):
    interval = 500_000_000
    learner = _make_learner(
        monkeypatch,
        tmp_path,
        target_entropy=0.01,
        target_entropy_schedule="adaptive",
        target_entropy_ema_decay=0.5,
        target_entropy_mean_threshold=1e-12,
        target_entropy_std_threshold=0.0,
        target_entropy_decay_factor=0.5,
        target_entropy_required_count=1,
        target_entropy_force_decay_steps=interval,
    )
    _fill_replay(learner)

    try:
        # Update/local counters reach the interval while global received agent steps do not.
        learner.global_step = interval
        learner.local_step = interval
        learner.agent_steps = 0
        learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, interval)
        assert learner.target_entropy_controller.state_dict()["forced_decay_count"] == 0

        learner.global_step = 0
        learner.local_step = 0
        learner.agent_steps = interval - 1
        learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
        assert learner.target_entropy_controller.state_dict()["forced_decay_count"] == 0

        learner.agent_steps = interval
        learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
        state = learner.target_entropy_controller.state_dict()
        assert state["target_entropy"] == pytest.approx(0.005)
        assert state["forced_decay_count"] == 1
        assert state["last_decay_agent_steps"] == interval
        assert abs(state["mean_entropy"] - state["target_entropy"]) > 1e-3
        assert state["variance"] > 0.0
        assert state["qualifying_count"] == 0
    finally:
        learner.close()


def test_adaptive_entropy_state_round_trips_in_learner_checkpoint(monkeypatch, tmp_path):
    config = {
        "target_entropy": 0.5,
        "target_entropy_schedule": "adaptive",
        "target_entropy_mean_threshold": 0.2,
        "target_entropy_std_threshold": 0.1,
        "target_entropy_required_count": 3,
    }
    source = _make_learner(monkeypatch, tmp_path / "source", **config)
    source.target_entropy_controller.observe(0.5)
    source.target_entropy_controller.observe(0.75)
    expected_state = source.target_entropy_controller.state_dict()
    source.save_checkpoint()

    restored = _make_learner(
        monkeypatch,
        tmp_path / "restored",
        replay=TensorReplay(8, (3,)),
        **config,
    )
    try:
        restored.load_training_state(tmp_path / "source" / "trainer_state.pt")
        assert restored.target_entropy_controller.state_dict() == expected_state
        source.target_entropy_controller.observe(0.5)
        restored.target_entropy_controller.observe(0.5)
        assert restored.target_entropy_controller.state_dict() == source.target_entropy_controller.state_dict()
    finally:
        source.close()
        restored.close()


def test_forced_decay_interval_round_trips_across_learner_checkpoint(monkeypatch, tmp_path):
    interval = 500_000_000
    config = {
        "target_entropy": 0.5,
        "target_entropy_schedule": "adaptive",
        "target_entropy_mean_threshold": 1e-12,
        "target_entropy_std_threshold": 0.0,
        "target_entropy_force_decay_steps": interval,
    }
    source = _make_learner(monkeypatch, tmp_path / "source", **config)
    source.agent_steps = interval
    source.target_entropy_controller.observe(0.9, agent_steps=interval)
    expected_state = source.target_entropy_controller.state_dict()
    source.save_checkpoint()

    restored = _make_learner(
        monkeypatch,
        tmp_path / "restored",
        replay=TensorReplay(8, (3,)),
        **config,
    )
    try:
        restored.load_training_state(tmp_path / "source" / "trainer_state.pt")
        assert restored.agent_steps == interval
        assert restored.target_entropy_controller.state_dict() == expected_state

        restored.target_entropy_controller.observe(0.9, agent_steps=interval + interval - 1)
        assert restored.target_entropy_controller.state_dict()["forced_decay_count"] == 1
        restored.target_entropy_controller.observe(0.9, agent_steps=interval + interval)
        resumed_state = restored.target_entropy_controller.state_dict()
        assert resumed_state["forced_decay_count"] == 2
        assert resumed_state["last_decay_agent_steps"] == interval + interval
    finally:
        source.close()
        restored.close()


def test_legacy_controller_v1_checkpoint_requires_forced_decay_disabled(monkeypatch, tmp_path):
    legacy_config = {
        "target_entropy": 0.5,
        "target_entropy_schedule": "adaptive",
        "target_entropy_force_decay_steps": None,
    }
    source = _make_learner(monkeypatch, tmp_path / "source", **legacy_config)
    source.target_entropy_controller.observe(0.5)
    source.save_checkpoint()
    checkpoint_path = tmp_path / "source" / "trainer_state.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    controller_state = checkpoint["target_entropy_controller_state_dict"]
    for key in ("force_decay_steps", "last_decay_agent_steps", "forced_decay_count"):
        controller_state.pop(key, None)
    controller_state["version"] = 1
    checkpoint["config"].pop("target_entropy_force_decay_steps", None)
    torch.save(checkpoint, checkpoint_path)

    restored_disabled = _make_learner(
        monkeypatch, tmp_path / "restored-disabled", **legacy_config,
    )
    restored_enabled = _make_learner(
        monkeypatch,
        tmp_path / "restored-enabled",
        target_entropy=0.5,
        target_entropy_schedule="adaptive",
        target_entropy_force_decay_steps=500_000_000,
    )
    try:
        restored_disabled.load_training_state(checkpoint_path)
        migrated_state = restored_disabled.target_entropy_controller.state_dict()
        assert migrated_state["version"] == 2
        assert migrated_state["forced_decay_count"] == 0
        assert migrated_state["last_decay_agent_steps"] == 0

        # Keep the saved config aligned so rejection comes from the v1 state
        # lacking the timer required by enabled force decay.
        checkpoint["config"]["target_entropy_force_decay_steps"] = 500_000_000
        torch.save(checkpoint, checkpoint_path)
        with pytest.raises(ValueError):
            restored_enabled.load_training_state(checkpoint_path)
    finally:
        source.close()
        restored_disabled.close()
        restored_enabled.close()


def test_checkpoint_rejects_forced_decay_timer_after_saved_agent_steps(monkeypatch, tmp_path):
    interval = 500_000_000
    config = {
        "target_entropy": 0.5,
        "target_entropy_schedule": "adaptive",
        "target_entropy_mean_threshold": 1e-12,
        "target_entropy_std_threshold": 0.0,
        "target_entropy_force_decay_steps": interval,
    }
    source = _make_learner(monkeypatch, tmp_path / "source", **config)
    source.agent_steps = interval
    source.target_entropy_controller.observe(0.9, agent_steps=interval)
    source.save_checkpoint()
    checkpoint_path = tmp_path / "source" / "trainer_state.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["agent_steps"] = interval - 1
    torch.save(checkpoint, checkpoint_path)

    restored = _make_learner(
        monkeypatch,
        tmp_path / "restored",
        replay=TensorReplay(8, (3,)),
        **config,
    )
    try:
        with pytest.raises(ValueError, match="timer exceeds saved agent_steps"):
            restored.load_training_state(checkpoint_path)
    finally:
        source.close()
        restored.close()


def test_legacy_constant_checkpoint_without_controller_state_loads(monkeypatch, tmp_path):
    source = _make_learner(monkeypatch, tmp_path / "source")
    source.save_checkpoint()
    checkpoint_path = tmp_path / "source" / "trainer_state.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in (
        "target_entropy_schedule",
        "target_entropy_ema_decay",
        "target_entropy_mean_threshold",
        "target_entropy_std_threshold",
        "target_entropy_decay_factor",
        "target_entropy_required_count",
    ):
        state["config"].pop(key, None)
    state.pop("target_entropy_controller_state_dict", None)
    torch.save(state, checkpoint_path)

    restored = _make_learner(monkeypatch, tmp_path / "restored")
    try:
        restored.load_training_state(checkpoint_path)
        assert restored.config["target_entropy_schedule"] == "constant"
        assert restored.target_entropy_controller is None
    finally:
        source.close()
        restored.close()


def test_adaptive_checkpoint_without_controller_state_is_rejected(monkeypatch, tmp_path):
    config = {"target_entropy": 0.5, "target_entropy_schedule": "adaptive"}
    source = _make_learner(monkeypatch, tmp_path / "source", **config)
    source.save_checkpoint()
    checkpoint_path = tmp_path / "source" / "trainer_state.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state.pop("target_entropy_controller_state_dict", None)
    torch.save(state, checkpoint_path)

    restored = _make_learner(monkeypatch, tmp_path / "restored", **config)
    try:
        with pytest.raises(ValueError, match="missing controller state"):
            restored.load_training_state(checkpoint_path)
    finally:
        source.close()
        restored.close()
