"""Semantic tests for cycle-cached discrete SAC actor Q values."""

import copy
from collections import Counter, defaultdict
import math
import os

import pytest
import torch
import torch.nn.functional as F

from pufferlib.sac_replay import TensorReplay
import project.PPO_SAC_Profiling.sac_variants as sac_variants
from project.PPO_SAC_Profiling.sac_variants import install_variant
from project.PPO_SAC_Profiling.test_fixed_targets import _manual_targets
from project.PPO_SAC_Profiling.test_sac_variants import (
    _TinyActor,
    _TinyCritic,
    _make_learner,
)


VARIANT = "fixed_targets_actor_q"


def _minimum_q_for_replay(learner):
    replay = learner.replay
    minimum_q = None
    with torch.no_grad():
        for start_idx in range(0, len(replay), learner.max_minibatch_size):
            end_idx = min(start_idx + learner.max_minibatch_size, len(replay))
            observations = replay.observations[start_idx:end_idx].to(
                learner.device, non_blocking=True,
            )
            with learner.amp_context:
                q1, q2 = learner.critic(observations)
            chunk_minimum_q = torch.minimum(q1.float(), q2.float())
            if minimum_q is None:
                minimum_q = torch.empty(
                    (len(replay), *chunk_minimum_q.shape[1:]),
                    device=learner.device,
                    dtype=torch.float32,
                )
            minimum_q[start_idx:end_idx].copy_(chunk_minimum_q)
    return minimum_q.detach()


def _actor_loss_oracle(learner, observations, cached_minimum_q):
    """Independent SAC actor objective with fixed Q and live policy/alpha."""
    logits = learner.policy_train(observations)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    probabilities = logprobs.exp()
    minimum_q = cached_minimum_q.float()
    loss = (
        probabilities * (learner.alpha * logprobs - minimum_q)
    ).sum(dim=-1).mean()
    entropy = -(probabilities * logprobs).sum(dim=-1).mean()
    return loss, entropy.detach()


def _replay_row_lookup(learner):
    return {
        tuple(observation.tolist()): row_index
        for row_index, observation in enumerate(
            learner.replay.observations[:len(learner.replay)].detach().cpu()
        )
    }


def _capture_and_check_actor_loss(monkeypatch, learner):
    """Capture row-aligned cached Q and check helper loss plus policy gradients."""
    original_helper = sac_variants._fixed_actor_q_policy_loss
    row_lookup = _replay_row_lookup(learner)
    captured = []

    def checked_helper(current_learner, observations, cached_minimum_q):
        assert cached_minimum_q.requires_grad is False
        assert cached_minimum_q.grad_fn is None
        loss, entropy = original_helper(
            current_learner, observations, cached_minimum_q,
        )
        expected_loss, expected_entropy = _actor_loss_oracle(
            current_learner, observations, cached_minimum_q,
        )
        torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(entropy, expected_entropy, rtol=1e-6, atol=1e-7)

        policy_parameters = tuple(current_learner.policy_train.parameters())
        actual_gradients = torch.autograd.grad(
            loss, policy_parameters, retain_graph=True,
        )
        expected_gradients = torch.autograd.grad(
            expected_loss, policy_parameters,
        )
        for actual_gradient, expected_gradient in zip(
                actual_gradients, expected_gradients):
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7,
            )

        row_indices = [
            row_lookup[tuple(row.tolist())]
            for row in observations.detach().cpu()
        ]
        captured.append((
            row_indices,
            cached_minimum_q.detach().cpu().clone(),
            float(current_learner.alpha.detach().cpu()),
        ))
        return loss, entropy

    monkeypatch.setattr(sac_variants, "_fixed_actor_q_policy_loss", checked_helper)
    return captured


def _attach_online_critic_counter(learner):
    counts = {"rows": 0, "calls": 0}

    def count_rows(_module, inputs, _output):
        counts["calls"] += 1
        counts["rows"] += int(inputs[0].shape[0])

    handle = learner.critic.register_forward_hook(count_rows)
    return counts, handle


def _assert_captured_q_follows_replay_rows(captured, expected_q, *, repeats):
    row_counts = Counter()
    for row_indices, cached_q, _alpha in captured:
        expected_batch_q = expected_q[
            torch.tensor(row_indices, dtype=torch.int64, device=expected_q.device)
        ].detach().cpu()
        torch.testing.assert_close(cached_q, expected_batch_q, rtol=0, atol=0)
        row_counts.update(row_indices)
    assert row_counts == Counter({row: repeats for row in range(expected_q.shape[0])})


def _make_gpu_replay_learner(actor_state, critic_state, *, update_epochs=3):
    learner = _make_learner(
        actor_state,
        critic_state,
        update_epochs=update_epochs,
        device="cuda",
        pin_memory=False,
    )
    cpu_replay = learner.replay
    gpu_replay = TensorReplay(
        cpu_replay.capacity,
        cpu_replay.observation_shape,
        device="cuda",
        pin_memory=False,
    )
    gpu_replay.add(*(
        getattr(cpu_replay, field_name)[:len(cpu_replay)].to("cuda")
        for field_name in (
            "observations", "actions", "rewards", "next_observations", "discounts",
        )
    ))
    learner.replay = gpu_replay
    return learner


def test_fixed_actor_q_reuses_detached_rows_and_matches_current_policy_alpha_loss_and_gradients(
        monkeypatch):
    torch.manual_seed(1103)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    learner = _make_learner(actor_state, critic_state, update_epochs=3)
    install_variant(learner, VARIANT)

    expected_q = _minimum_q_for_replay(learner)
    expected_targets = _manual_targets(learner)
    initial_policy = copy.deepcopy(learner.policy.state_dict())
    initial_critic = copy.deepcopy(learner.critic.state_dict())
    initial_target_critic = copy.deepcopy(learner.target_critic.state_dict())
    initial_alpha = learner.alpha.detach().clone()

    captured = _capture_and_check_actor_loss(monkeypatch, learner)
    critic_forward_counts, critic_hook = _attach_online_critic_counter(learner)
    updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )
    critic_hook.remove()

    replay_rows = len(learner.replay)
    epochs = learner.config["update_epochs"]
    expected_precompute_calls = math.ceil(replay_rows / learner.max_minibatch_size)
    assert updates == 6
    assert learner.fixed_cycle_actor_q.shape == (replay_rows, 3)
    assert learner.fixed_cycle_actor_q.device.type == "cpu"
    assert learner.fixed_cycle_actor_q.requires_grad is False
    assert learner.fixed_cycle_actor_q.grad_fn is None
    assert learner.fixed_actor_q_precompute_rows == replay_rows
    assert learner.fixed_actor_q_precompute_calls == expected_precompute_calls
    assert learner.actor_q_forward_rows == replay_rows
    assert learner.actor_q_forward_calls == expected_precompute_calls
    assert learner.fixed_actor_loss_rows == replay_rows * epochs
    torch.testing.assert_close(learner.fixed_cycle_actor_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(learner.fixed_cycle_targets, expected_targets, rtol=1e-6, atol=1e-7)

    _assert_captured_q_follows_replay_rows(captured, expected_q, repeats=epochs)
    expected_actor_microbatches_per_epoch = sum(
        math.ceil(logical_batch_count / learner.max_minibatch_size)
        for logical_batch_count in (4, 3)
    )
    assert len(captured) == epochs * expected_actor_microbatches_per_epoch
    alphas_used = [alpha for _indices, _q, alpha in captured]
    assert max(alphas_used) - min(alphas_used) > 1e-6
    assert not torch.equal(learner.alpha.detach(), initial_alpha)

    # One full-buffer Q precompute plus critic-loss forwards; actor training adds none.
    critic_loss_rows = replay_rows * epochs
    assert critic_forward_counts["rows"] == replay_rows + critic_loss_rows
    assert critic_forward_counts["calls"] == (
        expected_precompute_calls + epochs * 3
    )
    assert learner.learner_updates == updates
    assert learner.optimizer.state
    assert learner.critic_optimizer.state
    assert learner.alpha_optimizer.state
    assert any(
        not torch.equal(value, initial_policy[key])
        for key, value in learner.policy.state_dict().items()
    )
    assert any(
        not torch.equal(value, initial_critic[key])
        for key, value in learner.critic.state_dict().items()
    )
    assert any(
        not torch.equal(value, initial_target_critic[key])
        for key, value in learner.target_critic.state_dict().items()
    )


def test_fixed_actor_q_and_critic_targets_rebuild_all_rows_after_partial_replay_replacement():
    torch.manual_seed(1117)
    learner = _make_learner(
        _TinyActor().state_dict(), _TinyCritic().state_dict(), update_epochs=1,
    )
    install_variant(learner, VARIANT)

    learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )
    first_actor_q = learner.fixed_cycle_actor_q.detach().clone()
    first_targets = learner.fixed_cycle_targets.detach().clone()
    first_actor_q_cache = learner.fixed_cycle_actor_q
    first_target_cache = learner.fixed_cycle_targets

    replay = learner.replay
    replaced_observation = torch.tensor([[7.5, -3.0, 4.25]])
    replay.replace(
        torch.tensor([0]),
        replaced_observation,
        torch.tensor([2]),
        torch.tensor([1.3]),
        replaced_observation * -0.4,
        torch.tensor([0.97]),
    )
    expected_actor_q = _minimum_q_for_replay(learner)
    expected_targets = _manual_targets(learner)
    unchanged_rows = torch.tensor([1, 2, 3, 4, 5, 6])
    assert torch.equal(
        replay.observations[unchanged_rows],
        torch.arange(21, dtype=torch.float32).reshape(7, 3)[unchanged_rows] / 10,
    )

    learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 1,
    )

    assert learner.fixed_cycle_actor_q is not first_actor_q_cache
    assert learner.fixed_cycle_targets is not first_target_cache
    assert learner.fixed_actor_q_precompute_rows == len(replay)
    assert learner.fixed_target_precompute_rows == len(replay)
    torch.testing.assert_close(learner.fixed_cycle_actor_q, expected_actor_q, rtol=0, atol=0)
    torch.testing.assert_close(learner.fixed_cycle_targets, expected_targets, rtol=1e-6, atol=1e-7)
    assert torch.max(torch.abs(
        learner.fixed_cycle_actor_q[unchanged_rows] - first_actor_q[unchanged_rows]
    )).item() > 1e-6
    assert torch.max(torch.abs(
        learner.fixed_cycle_targets[unchanged_rows] - first_targets[unchanged_rows]
    )).item() > 1e-6


@pytest.mark.skipif(
    os.environ.get("SAC_PROFILE_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="set SAC_PROFILE_CUDA_TEST=1 on an idle CUDA device for GPU-replay actor-Q tests",
)
def test_cuda_gpu_replay_fixed_actor_q_matches_cache_oracle_and_skips_actor_critic_forward():
    torch.manual_seed(1123)
    torch.cuda.manual_seed_all(1123)
    learner = _make_gpu_replay_learner(
        _TinyActor().state_dict(), _TinyCritic().state_dict(), update_epochs=3,
    )
    install_variant(learner, VARIANT)
    expected_q = _minimum_q_for_replay(learner)
    expected_targets = _manual_targets(learner)
    captured = []
    row_lookup = _replay_row_lookup(learner)
    original_helper = sac_variants._fixed_actor_q_policy_loss

    def capture_q(current_learner, observations, cached_minimum_q):
        row_indices = [
            row_lookup[tuple(row.tolist())]
            for row in observations.detach().cpu()
        ]
        captured.append((
            row_indices,
            cached_minimum_q.detach().cpu().clone(),
            float(current_learner.alpha.detach().cpu()),
        ))
        return original_helper(current_learner, observations, cached_minimum_q)

    sac_variants._fixed_actor_q_policy_loss = capture_q
    critic_forward_counts, critic_hook = _attach_online_critic_counter(learner)
    try:
        updates = learner._train_sac_transition(
            defaultdict(float), lambda *_args, **_kwargs: None, 0,
        )
        torch.cuda.synchronize()
    finally:
        critic_hook.remove()
        sac_variants._fixed_actor_q_policy_loss = original_helper

    replay_rows = len(learner.replay)
    epochs = learner.config["update_epochs"]
    precompute_calls = math.ceil(replay_rows / learner.max_minibatch_size)
    assert updates == 6
    assert learner.replay.device.type == "cuda"
    assert learner.fixed_cycle_actor_q.device.type == "cuda"
    assert learner.fixed_cycle_actor_q.requires_grad is False
    assert learner.fixed_cycle_actor_q.grad_fn is None
    assert learner.fixed_actor_q_precompute_rows == replay_rows
    assert learner.actor_q_forward_rows == replay_rows
    assert learner.actor_q_forward_calls == precompute_calls
    assert learner.fixed_actor_loss_rows == replay_rows * epochs
    torch.testing.assert_close(
        learner.fixed_cycle_actor_q.cpu(), expected_q.cpu(), rtol=0, atol=0,
    )
    torch.testing.assert_close(
        learner.fixed_cycle_targets.cpu(), expected_targets.cpu(), rtol=1e-6, atol=1e-7,
    )
    _assert_captured_q_follows_replay_rows(captured, expected_q, repeats=epochs)
    assert critic_forward_counts["rows"] == replay_rows + replay_rows * epochs
    assert critic_forward_counts["calls"] == precompute_calls + epochs * 3
