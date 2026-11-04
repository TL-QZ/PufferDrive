import pytest
import torch

from pufferlib.sac_returns import compute_pql_targets


def _explicit_mixture(rewards, discounts, next_values, offsets, trace_lambda):
    targets = torch.empty_like(rewards, dtype=torch.float32)
    for segment_start, segment_end in zip(offsets[:-1], offsets[1:]):
        for row_index in range(segment_start, segment_end):
            horizon = segment_end - row_index
            n_step_returns = []
            for n_step in range(1, horizon + 1):
                endpoint_index = row_index + n_step - 1
                discounted_reward = torch.zeros((), dtype=torch.float32)
                cumulative_discount = torch.ones((), dtype=torch.float32)
                for reward_index in range(row_index, endpoint_index + 1):
                    discounted_reward += cumulative_discount * rewards[reward_index]
                    cumulative_discount *= discounts[reward_index]
                n_step_returns.append(
                    discounted_reward + cumulative_discount * next_values[endpoint_index]
                )

            mixture = torch.zeros((), dtype=torch.float32)
            for n_step, n_step_return in enumerate(n_step_returns, start=1):
                weight = (
                    trace_lambda ** (n_step - 1) * (1 - trace_lambda)
                    if n_step < horizon
                    else trace_lambda ** (horizon - 1)
                )
                mixture += weight * n_step_return
            targets[row_index] = mixture
    return targets


def test_pql_targets_match_independent_explicit_mixture_across_packed_segments():
    rewards = torch.tensor([1.0, -0.5, 3.0, 2.0, 0.25, -1.0, 4.0])
    discounts = torch.tensor([0.9, 0.8, 0.0, 0.7, 0.95, 0.6, 0.4])
    next_values = torch.tensor([5.0, 4.0, 99.0, -2.0, 3.0, 6.0, 8.0])
    offsets = torch.tensor([0, 3, 7], dtype=torch.int64)
    trace_lambda = 0.63

    actual = compute_pql_targets(rewards, discounts, next_values, offsets, trace_lambda)
    expected = _explicit_mixture(
        rewards, discounts, next_values, offsets.tolist(), trace_lambda
    )

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    assert not actual.requires_grad


@pytest.mark.parametrize("trace_lambda", [0.0, 1.0])
def test_pql_lambda_endpoints_match_one_step_or_full_available_suffix(trace_lambda):
    rewards = torch.tensor([1.0, 2.0, 3.0, -4.0])
    discounts = torch.tensor([0.5, 0.25, 0.75, 0.0])
    next_values = torch.tensor([10.0, 20.0, 30.0, 40.0])
    offsets = torch.tensor([0, 3, 4], dtype=torch.int64)

    actual = compute_pql_targets(rewards, discounts, next_values, offsets, trace_lambda)
    expected = _explicit_mixture(
        rewards, discounts, next_values, offsets.tolist(), trace_lambda
    )

    torch.testing.assert_close(actual, expected)


def test_single_row_segments_equal_one_step_sac_targets():
    rewards = torch.tensor([0.5, -2.0, 8.0])
    discounts = torch.tensor([0.9, 0.0, 0.25])
    next_values = torch.tensor([4.0, 500.0, -3.0])
    offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    actual = compute_pql_targets(rewards, discounts, next_values, offsets, 0.95)

    torch.testing.assert_close(actual, rewards + discounts * next_values)


def test_pql_targets_satisfy_cpu_gae_return_adapter_with_boundary_cuts():
    import pufferlib.pufferl

    rewards = torch.tensor([1.0, 0.25, -2.0, 3.0, 4.0])
    discounts = torch.tensor([0.9, 0.9, 0.0, 0.9, 0.9])
    next_values = torch.tensor([1.5, -0.5, 200.0, 2.0, 5.0])
    state_values = torch.tensor([0.5, 1.5, -0.5, 3.0, 2.0])
    offsets = [0, 3, 5]
    trace_lambda = 0.7
    gamma = 0.9

    targets = compute_pql_targets(
        rewards, discounts, next_values, torch.tensor(offsets), trace_lambda
    )
    actual_gae_advantages = torch.empty_like(state_values)
    for segment_start, segment_end in zip(offsets[:-1], offsets[1:]):
        segment_rewards = torch.cat(
            (torch.zeros(1), rewards[segment_start:segment_end])
        ).reshape(1, -1)
        segment_values = torch.cat(
            (state_values[segment_start:segment_end], next_values[segment_end - 1:segment_end])
        ).reshape(1, -1)
        segment_dones = torch.cat(
            (
                torch.zeros(1),
                (discounts[segment_start:segment_end] == 0).to(dtype=torch.float32),
            )
        ).reshape(1, -1)
        importance = torch.ones_like(segment_values)
        segment_advantages = torch.zeros_like(segment_values)
        torch.ops.pufferlib.compute_puff_advantage(
            segment_values,
            segment_rewards,
            segment_dones,
            importance,
            segment_advantages,
            gamma,
            trace_lambda,
            1.0,
            1.0,
        )
        actual_gae_advantages[segment_start:segment_end] = segment_advantages[0, :-1]

    torch.testing.assert_close(actual_gae_advantages, targets - state_values)


def test_pql_targets_are_detached_and_validate_packed_shapes():
    rewards = torch.tensor([1.0, 2.0], requires_grad=True)
    discounts = torch.tensor([0.9, 0.9], requires_grad=True)
    next_values = torch.tensor([3.0, 4.0], requires_grad=True)
    targets = compute_pql_targets(
        rewards, discounts, next_values, torch.tensor([0, 2]), 0.95
    )

    assert not targets.requires_grad
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_pql_targets(rewards.detach(), discounts.detach(), next_values.detach(), [0, 1, 1, 2], 0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pql_cuda_scan_matches_cpu():
    rewards = torch.tensor([1.0, -0.5, 3.0, 2.0, 0.25, -1.0, 4.0])
    discounts = torch.tensor([0.9, 0.8, 0.0, 0.7, 0.95, 0.6, 0.4])
    next_values = torch.tensor([5.0, 4.0, 99.0, -2.0, 3.0, 6.0, 8.0])
    offsets = torch.tensor([0, 3, 7], dtype=torch.int64)
    cpu_targets = compute_pql_targets(rewards, discounts, next_values, offsets, 0.73)
    cuda_targets = compute_pql_targets(
        rewards.cuda(), discounts.cuda(), next_values.cuda(), offsets, 0.73
    )

    torch.testing.assert_close(cuda_targets.cpu(), cpu_targets)
