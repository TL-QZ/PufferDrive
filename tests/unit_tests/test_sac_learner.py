import copy
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pufferlib import pufferl, pufferl_sac
from pufferlib.sac_replay import TensorReplay


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, observations):
        return self.linear(observations)


class _TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Linear(3, 3)
        self.q2 = nn.Linear(3, 3)

    def forward(self, observations):
        return self.q1(observations), self.q2(observations)


class _RecordingReplay(TensorReplay):
    def __init__(self, capacity, observation_shape):
        super().__init__(capacity, observation_shape)
        self.sampled_indices = []

    def get_batch(self, indices):
        self.sampled_indices.append(indices.clone())
        return super().get_batch(indices)


def _make_replay():
    replay = _RecordingReplay(7, (3,))
    values = torch.arange(21, dtype=torch.float32).reshape(7, 3) / 10
    replay.add(
        values,
        torch.tensor([0, 1, 2, 0, 1, 2, 0]),
        torch.tensor([0.2, -0.1, 0.5, 0.3, -0.2, 0.4, 0.1]),
        values * 0.7 + 0.15,
        torch.tensor([0.99, 0.99, 0.0, 0.99, 0.0, 0.99, 0.99]),
    )
    return replay


def _make_training_learner(replay, actor_state, critic_state, max_minibatch_size, max_grad_norm=None):
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.uncompiled_policy = _TinyActor()
    learner.uncompiled_policy.load_state_dict(actor_state)
    learner.policy = learner.uncompiled_policy
    learner.ddp_policy = learner.uncompiled_policy
    learner.policy_train = learner.policy
    learner.uncompiled_critic = _TinyCritic()
    learner.uncompiled_critic.load_state_dict(critic_state)
    learner.critic = learner.uncompiled_critic
    learner.ddp_critic = learner.uncompiled_critic
    learner.critic_train = learner.critic
    learner.uncompiled_target_critic = copy.deepcopy(learner.uncompiled_critic).requires_grad_(False).eval()
    learner.target_critic = learner.uncompiled_target_critic
    learner.amp_context = nullcontext()
    learner.log_alpha = nn.Parameter(torch.tensor(-1.2))
    learner.optimizer = torch.optim.Adam(learner.policy.parameters(), lr=0.003)
    learner.critic_optimizer = torch.optim.Adam(learner.critic.parameters(), lr=0.004)
    learner.alpha_optimizer = torch.optim.Adam([learner.log_alpha], lr=0.002)
    learner.config = {
        "update_epochs": 2,
        "target_entropy": 0.7,
        "tau": 0.15,
        "max_grad_norm": max_grad_norm,
    }
    learner.batch_size = replay.capacity
    learner.minibatch_size = 4
    learner.max_minibatch_size = max_minibatch_size
    learner.replay = replay
    learner.distributed = False
    learner.world_size = 1
    learner.learner_updates = 0
    learner.sampled_transitions = 0
    return learner


def _parameter_tensors(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def test_microbatches_match_logical_batch_updates_and_cover_each_epoch():
    torch.manual_seed(37)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    full_batch = _make_training_learner(_make_replay(), actor_state, critic_state, max_minibatch_size=4)
    microbatched = _make_training_learner(_make_replay(), actor_state, critic_state, max_minibatch_size=2)

    torch.manual_seed(101)
    full_losses = defaultdict(float)
    full_updates = full_batch._train_sac_transition(full_losses, lambda *_args, **_kwargs: None, 0)
    torch.manual_seed(101)
    micro_losses = defaultdict(float)
    micro_updates = microbatched._train_sac_transition(micro_losses, lambda *_args, **_kwargs: None, 0)

    assert full_updates == micro_updates == 4
    assert full_batch.learner_updates == microbatched.learner_updates == 4
    assert full_batch.sampled_transitions == microbatched.sampled_transitions == 14
    for full_values, micro_values in (
        (_parameter_tensors(full_batch.policy), _parameter_tensors(microbatched.policy)),
        (_parameter_tensors(full_batch.critic), _parameter_tensors(microbatched.critic)),
        (_parameter_tensors(full_batch.target_critic), _parameter_tensors(microbatched.target_critic)),
    ):
        for full_value, micro_value in zip(full_values, micro_values):
            torch.testing.assert_close(full_value, micro_value, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full_batch.log_alpha, microbatched.log_alpha, rtol=1e-5, atol=1e-6)
    assert full_losses.keys() == micro_losses.keys()
    for key in full_losses:
        assert micro_losses[key] == pytest.approx(full_losses[key], rel=1e-5, abs=1e-6)

    # Each logical update has critic calls followed by actor calls; collect critic rows.
    logical_batches = []
    for start_idx in range(0, len(microbatched.replay.sampled_indices), 4):
        critic_chunks = microbatched.replay.sampled_indices[start_idx:start_idx + 2]
        logical_batches.append(torch.cat(critic_chunks))
    assert [batch.numel() for batch in logical_batches] == [4, 3, 4, 3]
    for batch in logical_batches:
        assert torch.unique(batch).numel() == batch.numel()
    for epoch_start in (0, 2):
        epoch_indices = torch.cat(logical_batches[epoch_start:epoch_start + 2])
        torch.testing.assert_close(torch.sort(epoch_indices).values, torch.arange(7))


def test_gradient_clipping_runs_after_each_accumulated_update(monkeypatch):
    torch.manual_seed(11)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    learner = _make_training_learner(
        _make_replay(), actor_state, critic_state, max_minibatch_size=2, max_grad_norm=1e-5,
    )
    learner.config["update_epochs"] = 1
    learner.minibatch_size = 7

    actual_clip = torch.nn.utils.clip_grad_norm_
    clipped_norms = []

    def record_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        before_norm = torch.sqrt(sum(parameter.grad.square().sum() for parameter in parameters)).item()
        result = actual_clip(parameters, max_norm, *args, **kwargs)
        after_norm = torch.sqrt(sum(parameter.grad.square().sum() for parameter in parameters)).item()
        clipped_norms.append((max_norm, before_norm, after_norm))
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)

    assert len(clipped_norms) == 2
    for max_norm, before_norm, after_norm in clipped_norms:
        assert max_norm == 1e-5
        assert before_norm > max_norm
        assert after_norm <= max_norm * 1.001


def test_partial_replay_does_not_update_before_full_capacity():
    torch.manual_seed(13)
    replay = _RecordingReplay(7, (3,))
    partial = _make_replay()
    replay.add(*partial.get_batch(torch.arange(6)))
    learner = _make_training_learner(
        replay,
        copy.deepcopy(_TinyActor().state_dict()),
        copy.deepcopy(_TinyCritic().state_dict()),
        max_minibatch_size=4,
    )

    updates = learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)

    assert updates == 0
    assert learner.learner_updates == 0
    assert learner.sampled_transitions == 0
    assert replay.sampled_indices == []


@pytest.mark.parametrize(
    "config, agents_per_batch, expected_capacity, expected_minibatch",
    [
        ({"batch_size": "auto", "minibatch_size": "auto"}, 512, 512, 256),
        ({"batch_size": "auto", "minibatch_size": 384}, 256, 384, 384),
        ({"batch_size": 900, "minibatch_size": "auto"}, 512, 900, 256),
        ({"batch_size": 100, "minibatch_size": "auto"}, 512, 100, 100),
        ({"batch_size": 913, "minibatch_size": 347}, 512, 913, 347),
    ],
)
def test_auto_and_explicit_replay_sizes(config, agents_per_batch, expected_capacity, expected_minibatch):
    vecenv = SimpleNamespace(agents_per_batch=agents_per_batch)
    resolved = pufferl_sac.resolve_sac_batch_sizes(
        {"max_minibatch_size": 256, **config}, vecenv,
    )

    assert resolved["batch_size"] == expected_capacity
    assert resolved["minibatch_size"] == expected_minibatch


class _FakeVecEnv:
    def __init__(self):
        self.agents_per_batch = 4
        self.num_agents = 4
        self.single_observation_space = SimpleNamespace(shape=(3,))
        self.single_action_space = SimpleNamespace(n=3)

    def async_reset(self, seed):
        self.seed = seed


class _DummyUtilization:
    def stop(self):
        pass

    def join(self):
        pass


def _make_checkpoint_learner(monkeypatch, tmp_path, replay=None, vecenv=None, **config_updates):
    monkeypatch.setattr(pufferl, "Utilization", _DummyUtilization)
    monkeypatch.setattr(pufferl.PuffeRL, "print_dashboard", lambda self, clear=False: None)
    config = {
        "data_dir": str(tmp_path),
        "action_count": 3,
        "total_timesteps": 20,
        "batch_size": 8,
        "minibatch_size": 4,
        "max_minibatch_size": 256,
        **config_updates,
    }
    return pufferl_sac.PuffeSAC(
        config,
        _FakeVecEnv() if vecenv is None else vecenv,
        _TinyActor(),
        _TinyCritic(),
        replay if replay is not None else TensorReplay(config["batch_size"], (3,)),
    )


class _CollectVecEnv(_FakeVecEnv):
    def __init__(self):
        super().__init__()
        self.received = 0

    def recv(self):
        self.received += 1
        observation = torch.full((1, 3), float(self.received), dtype=torch.float32).numpy()
        return (
            observation,
            np.asarray([0.25], dtype=np.float32),
            np.asarray([False]),
            np.asarray([False]),
            [],
            np.asarray([0], dtype=np.int64),
            np.asarray([self.received > 1]),
        )

    def send(self, actions):
        self.last_actions = actions


def _old_version_five_checkpoint(learner):
    config = dict(learner.config)
    config["learning_starts"] = 2
    for key in (
        "max_minibatch_size", "optimizer", "adam_beta1", "adam_beta2", "adam_eps",
        "adam_weight_decay", "max_grad_norm",
    ):
        config.pop(key)
    return {
        "state_version": 5,
        "algorithm": "sac",
        "scope": "learner_only",
        "config": config,
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
        "epoch": 0,
        "valid_inserts": 0,
        "learner_updates": 0,
        "sampled_transitions": 0,
        "best_score": -float("inf"),
        "rng_state": pufferl.capture_rng_state(),
    }


def test_old_version_five_checkpoint_uses_legacy_optimizer_defaults(monkeypatch, tmp_path):
    learner = _make_checkpoint_learner(monkeypatch, tmp_path)
    checkpoint_path = tmp_path / "old_trainer_state.pt"
    torch.save(_old_version_five_checkpoint(learner), checkpoint_path)

    learner.load_training_state(checkpoint_path)

    assert learner.config["optimizer"] == "adam"
    assert learner.config["adam_eps"] == 1e-8
    assert learner.config["adam_weight_decay"] == 0.0
    assert learner.config["max_grad_norm"] is None


def test_cpu_bfloat16_autocasts_networks_and_keeps_sac_arithmetic_float32(monkeypatch, tmp_path):
    learner = _make_checkpoint_learner(
        monkeypatch,
        tmp_path,
        vecenv=_CollectVecEnv(),
        batch_size=8,
        total_timesteps=2,
        amp=True,
        precision="bfloat16",
    )
    network_dtypes = []
    handles = [
        learner.uncompiled_policy.linear.register_forward_hook(
            lambda _module, _inputs, output: network_dtypes.append(output.dtype)
        ),
        learner.uncompiled_critic.q1.register_forward_hook(
            lambda _module, _inputs, output: network_dtypes.append(output.dtype)
        ),
        learner.uncompiled_target_critic.q1.register_forward_hook(
            lambda _module, _inputs, output: network_dtypes.append(output.dtype)
        ),
    ]

    learner.evaluate()
    observations, actions, rewards, next_observations, discounts = learner.replay.get_batch(
        torch.tensor([0], dtype=torch.int64)
    )
    targets = learner._compute_targets(rewards, next_observations, discounts)
    critic_loss, q1, _ = learner._critic_loss(observations, actions, targets)
    policy_loss, entropy = learner._policy_loss(observations)
    for handle in handles:
        handle.remove()

    assert torch.bfloat16 in network_dtypes
    assert all(parameter.dtype == torch.float32 for parameter in learner.uncompiled_policy.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in learner.uncompiled_critic.parameters())
    assert targets.dtype == torch.float32
    assert critic_loss.dtype == q1.dtype == policy_loss.dtype == entropy.dtype == torch.float32
    assert torch.isfinite(targets).all() and torch.isfinite(critic_loss) and torch.isfinite(policy_loss)


def test_compiled_cpu_learner_trains_tail_batches_and_round_trips_uncompiled_state(
        monkeypatch, tmp_path):
    learner_config = {
        "batch_size": 7,
        "minibatch_size": 7,
        "max_minibatch_size": 3,
        "update_epochs": 2,
    }
    torch.manual_seed(23)
    learner = _make_checkpoint_learner(
        monkeypatch,
        tmp_path,
        compile=True,
        replay=_make_replay(),
        **learner_config,
    )
    torch.manual_seed(23)
    eager = _make_checkpoint_learner(
        monkeypatch,
        tmp_path,
        compile=False,
        replay=_make_replay(),
        **learner_config,
    )

    torch.manual_seed(101)
    compiled_losses = defaultdict(float)
    updates = learner._train_sac_transition(compiled_losses, lambda *_args, **_kwargs: None, 0)
    torch.manual_seed(101)
    eager_losses = defaultdict(float)
    eager_updates = eager._train_sac_transition(eager_losses, lambda *_args, **_kwargs: None, 0)

    assert updates == eager_updates == learner.learner_updates == eager.learner_updates == 2
    assert learner.sampled_transitions == eager.sampled_transitions == 14
    assert [indices.numel() for indices in learner.replay.sampled_indices] == [3, 3, 1] * 4
    for compiled_model, eager_model in (
        (learner.uncompiled_policy, eager.uncompiled_policy),
        (learner.uncompiled_critic, eager.uncompiled_critic),
        (learner.uncompiled_target_critic, eager.uncompiled_target_critic),
    ):
        for compiled_parameter, eager_parameter in zip(compiled_model.parameters(), eager_model.parameters()):
            torch.testing.assert_close(compiled_parameter, eager_parameter, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(learner.log_alpha, eager.log_alpha, rtol=1e-5, atol=1e-6)
    assert compiled_losses.keys() == eager_losses.keys()
    for key, value in compiled_losses.items():
        assert value == pytest.approx(eager_losses[key], rel=1e-5, abs=1e-6)

    checkpoint_path = learner.save_checkpoint()
    actor_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    trainer_state = torch.load(tmp_path / "trainer_state.pt", map_location="cpu", weights_only=False)
    assert not any("_orig_mod" in key for key in actor_state)
    assert not any("_orig_mod" in key for key in trainer_state["critic_state_dict"])

    restored = _make_checkpoint_learner(
        monkeypatch,
        tmp_path,
        replay=TensorReplay(7, (3,)),
        compile=False,
        **learner_config,
    )
    restored.load_training_state(tmp_path / "trainer_state.pt")
    for source, loaded in (
        (learner.uncompiled_policy, restored.uncompiled_policy),
        (learner.uncompiled_critic, restored.uncompiled_critic),
        (learner.uncompiled_target_critic, restored.uncompiled_target_critic),
    ):
        for source_value, loaded_value in zip(source.parameters(), loaded.parameters()):
            torch.testing.assert_close(source_value, loaded_value)


@pytest.mark.parametrize(
    "config_updates, mismatched_key",
    [({"optimizer": "adamw"}, "optimizer"), ({"adam_eps": 1e-6}, "adam_eps")],
)
def test_old_version_five_checkpoint_rejects_new_optimizer_settings(
        monkeypatch, tmp_path, config_updates, mismatched_key):
    learner = _make_checkpoint_learner(monkeypatch, tmp_path, **config_updates)
    checkpoint_path = tmp_path / "old_trainer_state.pt"
    torch.save(_old_version_five_checkpoint(learner), checkpoint_path)

    with pytest.raises(ValueError, match=f"Checkpoint config mismatch: {mismatched_key}"):
        learner.load_training_state(checkpoint_path)


@pytest.mark.parametrize(
    "optimizer_name, expected_type",
    [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)],
)
def test_actor_critic_optimizer_selection_and_alpha_adam(monkeypatch, tmp_path, optimizer_name, expected_type):
    learner = _make_checkpoint_learner(
        monkeypatch,
        tmp_path,
        learning_rate=0.001,
        critic_learning_rate=0.002,
        alpha_learning_rate=0.003,
        optimizer=optimizer_name,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_eps=1e-6,
        adam_weight_decay=0.12,
    )

    assert type(learner.optimizer) is expected_type
    assert type(learner.critic_optimizer) is expected_type
    assert isinstance(learner.alpha_optimizer, torch.optim.Adam)
    assert learner.optimizer.param_groups[0]["lr"] == 0.001
    assert learner.critic_optimizer.param_groups[0]["lr"] == 0.002
    assert learner.alpha_optimizer.param_groups[0]["lr"] == 0.003
    for optimizer in (learner.optimizer, learner.critic_optimizer):
        params = optimizer.param_groups[0]
        assert params["betas"] == (0.8, 0.95)
        assert params["eps"] == 1e-6
        assert params["weight_decay"] == 0.12


def test_muon_optimizer_runs_actor_and_critic_steps_with_installed_heavyball(monkeypatch, tmp_path):
    heavyball = pytest.importorskip("heavyball")
    muon_type = getattr(heavyball, "ForeachMuon", None) or getattr(heavyball, "Muon", None)
    if muon_type is None:
        pytest.skip("installed heavyball has no supported Muon optimizer")
    learner = _make_checkpoint_learner(monkeypatch, tmp_path, optimizer="muon")

    assert isinstance(learner.optimizer, muon_type)
    assert isinstance(learner.critic_optimizer, muon_type)
    assert isinstance(learner.alpha_optimizer, torch.optim.Adam)

    actor_before = _parameter_tensors(learner.policy)
    critic_before = _parameter_tensors(learner.critic)
    observations = torch.randn(5, 3)
    learner.optimizer.zero_grad(set_to_none=True)
    learner.policy(observations).square().mean().backward()
    learner.optimizer.step()
    learner.critic_optimizer.zero_grad(set_to_none=True)
    q1, q2 = learner.critic(observations)
    (q1.square().mean() + q2.square().mean()).backward()
    learner.critic_optimizer.step()

    assert any(not torch.equal(before, after) for before, after in zip(actor_before, learner.policy.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(critic_before, learner.critic.parameters()))
