"""Semantic tests for fixed-within-cycle discrete SAC Bellman targets."""

import copy
from collections import Counter, defaultdict
import math
import os

import pytest
import torch

from project.PPO_SAC_Profiling.sac_variants import install_variant
from project.PPO_SAC_Profiling.test_sac_variants import (
    _TinyActor,
    _TinyCritic,
    _assert_learner_state_equal,
    _make_learner,
)


FIXED_VARIANTS = ("fixed_targets", "fixed_targets_cached")


def _manual_targets(learner, *, target_policy=None):
    """Compute discrete soft Bellman targets with scalar Python arithmetic."""
    replay = learner.replay
    with torch.no_grad():
        next_observations = replay.next_observations[:len(replay)].to(
            learner.device, non_blocking=True,
        )
        bootstrap_policy = learner.policy if target_policy is None else target_policy
        logits = bootstrap_policy(next_observations)
        target_q1, target_q2 = learner.target_critic(next_observations)
        alpha = float(learner.alpha.detach().cpu())

    expected = []
    for row_index in range(len(replay)):
        row_logits = [float(value) for value in logits[row_index].cpu().tolist()]
        max_logit = max(row_logits)
        exp_logits = [math.exp(value - max_logit) for value in row_logits]
        normalizer = sum(exp_logits)
        probabilities = [value / normalizer for value in exp_logits]
        log_probabilities = [
            value - max_logit - math.log(normalizer) for value in row_logits
        ]
        soft_value = 0.0
        for action_index, probability in enumerate(probabilities):
            min_q = min(
                float(target_q1[row_index, action_index].cpu()),
                float(target_q2[row_index, action_index].cpu()),
            )
            if learner.config["critic_entropy_in_target"]:
                min_q -= alpha * log_probabilities[action_index]
            soft_value += probability * min_q
        reward = float(replay.rewards[row_index].cpu())
        discount = float(replay.discounts[row_index].cpu())
        expected.append(reward + discount * soft_value)
    return torch.tensor(expected, dtype=torch.float32, device=learner.device)


def _row_indices(learner):
    return {
        tuple(observation.tolist()): row_index
        for row_index, observation in enumerate(learner.replay.observations[:len(learner.replay)])
    }


def _capture_critic_targets(learner):
    seen = []
    lookup = _row_indices(learner)
    original_critic_loss = learner._critic_loss

    def capture(observations, actions, targets):
        assert targets.requires_grad is False
        assert targets.grad_fn is None
        row_indices = [lookup[tuple(row.tolist())] for row in observations.detach().cpu()]
        seen.append((row_indices, targets.detach().cpu().clone()))
        return original_critic_loss(observations, actions, targets)

    learner._critic_loss = capture
    return seen


def _assert_targets_follow_row_indices(seen, expected, *, repeats):
    counts = Counter()
    for row_indices, targets in seen:
        torch.testing.assert_close(
            targets,
            expected[torch.tensor(row_indices, dtype=torch.int64, device=expected.device)].cpu(),
            rtol=1e-6,
            atol=1e-7,
        )
        counts.update(row_indices)
    assert counts == Counter({row_index: repeats for row_index in range(expected.numel())})


@pytest.mark.parametrize("variant", FIXED_VARIANTS)
def test_fixed_targets_match_manual_discrete_soft_bellman_equation(variant):
    torch.manual_seed(701)
    learner = _make_learner(
        _TinyActor().state_dict(), _TinyCritic().state_dict(), update_epochs=3,
    )
    install_variant(learner, variant)

    expected = _manual_targets(learner)
    seen = _capture_critic_targets(learner)
    torch.manual_seed(809)
    updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0, update_policy=False,
    )

    assert updates == 6
    _assert_targets_follow_row_indices(seen, expected, repeats=3)
    for terminal_row in (2, 4):
        assert learner.replay.discounts[terminal_row] == 0
        assert expected[terminal_row] == learner.replay.rewards[terminal_row]

    assert learner.fixed_cycle_targets.shape == (len(learner.replay),)
    assert learner.fixed_cycle_targets.requires_grad is False
    assert learner.fixed_cycle_targets.grad_fn is None
    assert learner.fixed_target_precompute_rows == len(learner.replay)
    assert learner.fixed_target_precompute_calls == math.ceil(
        len(learner.replay) / learner.max_minibatch_size
    )
    torch.testing.assert_close(learner.fixed_cycle_targets, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("variant", FIXED_VARIANTS)
def test_fixed_targets_are_rebuilt_per_cycle_and_intentionally_diverge_after_updates(variant):
    torch.manual_seed(719)
    learner = _make_learner(
        _TinyActor().state_dict(), _TinyCritic().state_dict(), update_epochs=3,
    )
    install_variant(learner, variant)
    initial_policy_state = copy.deepcopy(learner.uncompiled_policy.state_dict())
    initial_critic_state = copy.deepcopy(learner.uncompiled_critic.state_dict())
    initial_target_state = copy.deepcopy(learner.uncompiled_target_critic.state_dict())
    initial_alpha = learner.log_alpha.detach().clone()

    first_cycle_expected = _manual_targets(learner)
    first_seen = _capture_critic_targets(learner)
    torch.manual_seed(907)
    first_updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )
    assert first_updates == 6
    _assert_targets_follow_row_indices(first_seen, first_cycle_expected, repeats=3)
    first_cache = learner.fixed_cycle_targets
    first_cache_snapshot = first_cache.detach().clone()

    refreshed_targets = _manual_targets(learner)
    assert torch.max(torch.abs(refreshed_targets - first_cache_snapshot)).item() > 1e-5
    assert learner.learner_updates == 6
    assert learner.optimizer.state
    assert learner.critic_optimizer.state
    assert learner.alpha_optimizer.state
    assert not torch.equal(learner.log_alpha.detach(), initial_alpha)
    assert any(
        not torch.equal(value, initial_policy_state[key])
        for key, value in learner.uncompiled_policy.state_dict().items()
    )
    assert any(
        not torch.equal(value, initial_critic_state[key])
        for key, value in learner.uncompiled_critic.state_dict().items()
    )
    assert any(
        not torch.equal(value, initial_target_state[key])
        for key, value in learner.uncompiled_target_critic.state_dict().items()
    )

    second_seen = _capture_critic_targets(learner)
    torch.manual_seed(911)
    second_updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 1,
    )
    assert second_updates == 6
    _assert_targets_follow_row_indices(second_seen, refreshed_targets, repeats=3)
    assert learner.fixed_cycle_targets is not first_cache
    torch.testing.assert_close(
        learner.fixed_cycle_targets, refreshed_targets, rtol=1e-6, atol=1e-7,
    )
    assert learner.fixed_target_precompute_rows == len(learner.replay)


def test_cached_and_uncached_fixed_target_variants_have_identical_updates():
    torch.manual_seed(727)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    uncached = _make_learner(actor_state, critic_state)
    cached = _make_learner(actor_state, critic_state)
    install_variant(uncached, "fixed_targets")
    install_variant(cached, "fixed_targets_cached")

    torch.manual_seed(919)
    uncached_losses = defaultdict(float)
    uncached_updates = uncached._train_sac_transition(
        uncached_losses, lambda *_args, **_kwargs: None, 0,
    )
    torch.manual_seed(919)
    cached_losses = defaultdict(float)
    cached_updates = cached._train_sac_transition(
        cached_losses, lambda *_args, **_kwargs: None, 0,
    )

    assert uncached_updates == cached_updates == 4
    assert dict(uncached_losses) == dict(cached_losses)
    _assert_learner_state_equal(cached, uncached)
    torch.testing.assert_close(
        cached.fixed_cycle_targets, uncached.fixed_cycle_targets, rtol=0, atol=0,
    )


def test_fixed_target_is_equivalent_to_stock_for_one_frozen_bootstrap_update():
    torch.manual_seed(733)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    stock = _make_learner(actor_state, critic_state, update_epochs=1)
    fixed = _make_learner(actor_state, critic_state, update_epochs=1)
    stock.minibatch_size = fixed.minibatch_size = len(stock.replay)
    install_variant(stock, "stock")
    install_variant(fixed, "fixed_targets")

    torch.manual_seed(929)
    stock_losses = defaultdict(float)
    stock_updates = stock._train_sac_transition(
        stock_losses, lambda *_args, **_kwargs: None, 0, update_policy=False,
    )
    torch.manual_seed(929)
    fixed_losses = defaultdict(float)
    fixed_updates = fixed._train_sac_transition(
        fixed_losses, lambda *_args, **_kwargs: None, 0, update_policy=False,
    )

    assert stock_updates == fixed_updates == 1
    assert dict(stock_losses) == dict(fixed_losses)
    _assert_learner_state_equal(fixed, stock)


@pytest.mark.parametrize("target_policy_enabled", [False, True])
@pytest.mark.parametrize("critic_entropy_in_target", [False, True])
def test_manual_oracle_honors_target_policy_and_entropy_switch(
        target_policy_enabled, critic_entropy_in_target):
    torch.manual_seed(739)
    learner = _make_learner(
        _TinyActor().state_dict(), _TinyCritic().state_dict(), update_epochs=1,
    )
    learner.config["critic_entropy_in_target"] = critic_entropy_in_target
    install_variant(learner, "fixed_targets")

    target_policy = None
    if target_policy_enabled:
        target_policy = _TinyActor()
        with torch.no_grad():
            target_policy.linear.weight.add_(0.35)
            target_policy.linear.bias.sub_(0.2)
        target_policy.requires_grad_(False).eval()
    expected = _manual_targets(learner, target_policy=target_policy)
    seen = _capture_critic_targets(learner)
    torch.manual_seed(937)
    updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
        update_policy=False, target_policy=target_policy,
    )

    assert updates == 2
    _assert_targets_follow_row_indices(seen, expected, repeats=1)
    torch.testing.assert_close(learner.fixed_cycle_targets, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(
    os.environ.get("SAC_PROFILE_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="set SAC_PROFILE_CUDA_TEST=1 on an idle CUDA device for pinned-replay fixed-target tests",
)
@pytest.mark.parametrize("variant", FIXED_VARIANTS)
def test_cuda_fixed_targets_match_manual_oracle_and_reuse_rows_across_passes(variant):
    torch.manual_seed(743)
    torch.cuda.manual_seed_all(743)
    learner = _make_learner(
        _TinyActor().state_dict(), _TinyCritic().state_dict(),
        update_epochs=3, device="cuda", pin_memory=True,
    )
    install_variant(learner, variant)

    expected = _manual_targets(learner)
    seen = _capture_critic_targets(learner)
    torch.manual_seed(941)
    torch.cuda.manual_seed_all(941)
    updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )
    torch.cuda.synchronize()

    assert updates == 6
    _assert_targets_follow_row_indices(seen, expected, repeats=3)
    assert learner.fixed_cycle_targets.device.type == "cuda"
    assert learner.fixed_cycle_targets.requires_grad is False
    assert learner.fixed_cycle_targets.grad_fn is None
    assert learner.fixed_target_precompute_rows == len(learner.replay)
    assert learner.fixed_target_precompute_calls == math.ceil(
        len(learner.replay) / learner.max_minibatch_size
    )
    torch.testing.assert_close(
        learner.fixed_cycle_targets.cpu(), expected.cpu(), rtol=1e-6, atol=1e-7,
    )


@pytest.mark.skipif(
    os.environ.get("SAC_PROFILE_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="set SAC_PROFILE_CUDA_TEST=1 on an idle CUDA device for pinned-replay fixed-target tests",
)
def test_cuda_fixed_target_cached_and_uncached_variants_have_identical_updates():
    torch.manual_seed(751)
    torch.cuda.manual_seed_all(751)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    uncached = _make_learner(actor_state, critic_state, device="cuda", pin_memory=True)
    cached = _make_learner(actor_state, critic_state, device="cuda", pin_memory=True)
    install_variant(uncached, "fixed_targets")
    install_variant(cached, "fixed_targets_cached")

    torch.manual_seed(953)
    torch.cuda.manual_seed_all(953)
    uncached_losses = defaultdict(float)
    uncached_updates = uncached._train_sac_transition(
        uncached_losses, lambda *_args, **_kwargs: None, 0,
    )
    torch.cuda.synchronize()
    torch.manual_seed(953)
    torch.cuda.manual_seed_all(953)
    cached_losses = defaultdict(float)
    cached_updates = cached._train_sac_transition(
        cached_losses, lambda *_args, **_kwargs: None, 0,
    )
    torch.cuda.synchronize()

    assert uncached_updates == cached_updates == 4
    assert dict(uncached_losses) == dict(cached_losses)
    _assert_learner_state_equal(cached, uncached)
    torch.testing.assert_close(
        cached.fixed_cycle_targets, uncached.fixed_cycle_targets, rtol=0, atol=0,
    )
