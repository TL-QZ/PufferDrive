"""Synthetic checks for censored soft-return calculations."""

import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.returns import compute_returns


def _scalar_oracle(
    rewards,
    log_probs,
    valid,
    terminals,
    truncations,
    next_soft_values,
    gamma,
    alpha,
):
    """Forward-scan oracle independent of the implementation's recurrence."""
    shape = rewards.shape
    outputs = {
        "reward_partial": np.full(shape, np.nan),
        "soft_partial": np.full(shape, np.nan),
        "entropy_contribution": np.full(shape, np.nan),
        "censored": np.zeros(shape, dtype=bool),
        "truncation_endpoint": np.zeros(shape, dtype=bool),
        "observed_horizon": np.zeros(shape, dtype=np.int64),
        "tail_discount": np.full(shape, np.nan),
        "bootstrap_value": np.full(shape, np.nan),
        "bootstrap_tail": np.full(shape, np.nan),
        "soft_bootstrapped": np.full(shape, np.nan),
    }

    for agent_idx in range(shape[1]):
        for start_idx in range(shape[0]):
            if not valid[start_idx, agent_idx]:
                continue

            trajectory_indices = []
            row_idx = start_idx
            terminal_endpoint = False
            truncation_endpoint = False
            while True:
                trajectory_indices.append(row_idx)
                if terminals[row_idx, agent_idx]:
                    terminal_endpoint = True
                    break
                if truncations[row_idx, agent_idx]:
                    truncation_endpoint = True
                    break
                successor_idx = row_idx + 1
                if successor_idx >= shape[0] or not valid[successor_idx, agent_idx]:
                    break
                row_idx = successor_idx

            horizon = len(trajectory_indices)
            discounted_reward = sum(
                gamma**offset * rewards[transition_idx, agent_idx]
                for offset, transition_idx in enumerate(trajectory_indices)
            )
            discounted_entropy = sum(
                gamma**offset
                * -alpha
                * log_probs[transition_idx, agent_idx]
                for offset, transition_idx in enumerate(trajectory_indices[1:], start=1)
            )
            censored_endpoint = not terminal_endpoint
            tail_discount = gamma**horizon if censored_endpoint else 0.0
            bootstrap_value = np.nan
            if censored_endpoint:
                bootstrap_value = next_soft_values[row_idx, agent_idx]
                bootstrap_tail = (
                    0.0
                    if tail_discount == 0.0
                    else tail_discount * bootstrap_value
                )
            else:
                bootstrap_tail = 0.0

            soft_partial = discounted_reward + discounted_entropy
            outputs["reward_partial"][start_idx, agent_idx] = discounted_reward
            outputs["soft_partial"][start_idx, agent_idx] = soft_partial
            outputs["entropy_contribution"][start_idx, agent_idx] = discounted_entropy
            outputs["censored"][start_idx, agent_idx] = censored_endpoint
            outputs["truncation_endpoint"][start_idx, agent_idx] = truncation_endpoint
            outputs["observed_horizon"][start_idx, agent_idx] = horizon
            outputs["tail_discount"][start_idx, agent_idx] = tail_discount
            outputs["bootstrap_value"][start_idx, agent_idx] = bootstrap_value
            outputs["bootstrap_tail"][start_idx, agent_idx] = bootstrap_tail
            outputs["soft_bootstrapped"][start_idx, agent_idx] = (
                soft_partial + bootstrap_tail
            )

    return outputs


def _assert_matches_oracle(arrays, expected):
    for name, expected_values in expected.items():
        actual_values = getattr(arrays, name)
        if expected_values.dtype == np.bool_ or np.issubdtype(expected_values.dtype, np.integer):
            np.testing.assert_array_equal(actual_values, expected_values, err_msg=name)
        else:
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
                err_msg=name,
            )


def test_one_step_truncation_bootstraps_endpoint_soft_value_without_initial_entropy():
    result = compute_returns(
        rewards=np.array([[2.0]]),
        log_probs=np.array([[-3.0]]),
        valid=np.array([[True]]),
        terminals=np.array([[False]]),
        truncations=np.array([[True]]),
        next_soft_values=np.array([[10.0]]),
        gamma=0.5,
        alpha=0.2,
    )

    assert result.reward_partial[0, 0] == 2.0
    assert result.soft_partial[0, 0] == 2.0
    assert result.entropy_contribution[0, 0] == 0.0
    assert result.censored[0, 0]
    assert result.truncation_endpoint[0, 0]
    assert result.observed_horizon[0, 0] == 1
    assert result.tail_discount[0, 0] == 0.5
    assert result.bootstrap_value[0, 0] == 10.0
    assert result.bootstrap_tail[0, 0] == 5.0
    assert result.soft_bootstrapped[0, 0] == 7.0


def test_terminal_takes_precedence_over_truncation_and_ignores_nan_endpoint_value():
    result = compute_returns(
        rewards=np.array([[4.0]]),
        log_probs=np.array([[-0.7]]),
        valid=np.array([[True]]),
        terminals=np.array([[True]]),
        truncations=np.array([[True]]),
        next_soft_values=np.array([[np.nan]]),
        gamma=0.99,
        alpha=0.3,
    )

    assert result.reward_partial[0, 0] == 4.0
    assert result.soft_partial[0, 0] == 4.0
    assert result.entropy_contribution[0, 0] == 0.0
    assert not result.censored[0, 0]
    assert not result.truncation_endpoint[0, 0]
    assert result.observed_horizon[0, 0] == 1
    assert result.tail_discount[0, 0] == 0.0
    assert np.isnan(result.bootstrap_value[0, 0])
    assert result.bootstrap_tail[0, 0] == 0.0
    assert result.soft_bootstrapped[0, 0] == 4.0


def test_truncation_stops_before_next_episode_and_bootstrap_includes_no_sampled_tail_entropy():
    result = compute_returns(
        rewards=np.array([[1.0], [100.0]]),
        log_probs=np.array([[-0.5], [-8.0]]),
        valid=np.array([[True], [True]]),
        terminals=np.array([[False], [True]]),
        truncations=np.array([[True], [False]]),
        next_soft_values=np.array([[6.0], [np.nan]]),
        gamma=0.5,
        alpha=0.2,
    )

    assert result.reward_partial[0, 0] == 1.0
    assert result.soft_partial[0, 0] == 1.0
    assert result.bootstrap_tail[0, 0] == 3.0
    assert result.soft_bootstrapped[0, 0] == 4.0
    assert result.truncation_endpoint[0, 0]
    assert result.reward_partial[1, 0] == 100.0
    assert not result.censored[1, 0]


def test_last_sampled_action_entropy_is_counted_but_bootstrap_endpoint_entropy_is_not_sampled_twice():
    result = compute_returns(
        rewards=np.array([[1.0], [2.0]]),
        log_probs=np.array([[-5.0], [-0.5]]),
        valid=np.array([[True], [True]]),
        terminals=np.array([[False], [False]]),
        truncations=np.array([[False], [False]]),
        next_soft_values=np.array([[np.nan], [4.0]]),
        gamma=0.9,
        alpha=0.2,
    )

    expected_partial = 1.0 + 0.9 * (2.0 + 0.2 * 0.5)
    assert result.reward_partial[0, 0] == pytest.approx(2.8)
    assert result.entropy_contribution[0, 0] == pytest.approx(0.09)
    assert result.soft_partial[0, 0] == pytest.approx(expected_partial)
    assert result.tail_discount[0, 0] == pytest.approx(0.9**2)
    assert result.bootstrap_tail[0, 0] == pytest.approx((0.9**2) * 4.0)
    assert result.soft_bootstrapped[0, 0] == pytest.approx(expected_partial + (0.9**2) * 4.0)


def test_invalid_gap_is_a_cut_and_later_valid_rows_are_not_bridged():
    result = compute_returns(
        rewards=np.array([[1.0], [np.nan], [100.0]]),
        log_probs=np.array([[-0.1], [np.nan], [-0.2]]),
        valid=np.array([[True], [False], [True]]),
        terminals=np.array([[False], [False], [True]]),
        truncations=np.zeros((3, 1), dtype=bool),
        next_soft_values=np.array([[5.0], [np.nan], [np.nan]]),
        gamma=0.5,
        alpha=0.1,
    )

    assert result.reward_partial[0, 0] == 1.0
    assert result.censored[0, 0]
    assert not result.truncation_endpoint[0, 0]
    assert result.tail_discount[0, 0] == 0.5
    assert result.soft_bootstrapped[0, 0] == 3.5
    assert np.isnan(result.reward_partial[1, 0])
    assert not result.censored[1, 0]
    assert result.observed_horizon[1, 0] == 0
    assert result.reward_partial[2, 0] == 100.0
    assert not result.censored[2, 0]


def test_mixed_agent_streams_match_forward_scan_oracle():
    rewards = np.arange(24, dtype=np.float64).reshape(6, 4) / 10.0
    log_probs = -np.arange(24, dtype=np.float64).reshape(6, 4) / 20.0
    valid = np.ones((6, 4), dtype=bool)
    valid[5, 1] = False
    valid[2, 2] = False
    rewards[5, 1] = np.nan
    log_probs[5, 1] = np.nan
    rewards[2, 2] = np.nan
    log_probs[2, 2] = np.nan
    terminals = np.zeros((6, 4), dtype=bool)
    terminals[3, 0] = True
    terminals[4, 1] = True
    terminals[5, 2] = True
    truncations = np.zeros((6, 4), dtype=bool)
    truncations[2, 1] = True
    next_soft_values = np.full((6, 4), np.nan)
    next_soft_values[2, 1] = 7.0
    next_soft_values[1, 2] = 8.0
    next_soft_values[5, 3] = 9.0

    expected = _scalar_oracle(
        rewards,
        log_probs,
        valid,
        terminals,
        truncations,
        next_soft_values,
        gamma=0.8,
        alpha=0.15,
    )
    actual = compute_returns(
        rewards,
        log_probs,
        valid,
        terminals,
        truncations,
        next_soft_values,
        gamma=0.8,
        alpha=0.15,
    )

    _assert_matches_oracle(actual, expected)


def test_zero_discount_makes_missing_censored_tail_determinate():
    result = compute_returns(
        rewards=np.array([[3.0], [5.0]]),
        log_probs=np.array([[-1.0], [-2.0]]),
        valid=np.array([[True], [True]]),
        terminals=np.zeros((2, 1), dtype=bool),
        truncations=np.zeros((2, 1), dtype=bool),
        next_soft_values=np.array([[np.nan], [np.nan]]),
        gamma=0.0,
        alpha=0.5,
    )

    np.testing.assert_array_equal(result.reward_partial[:, 0], [3.0, 5.0])
    np.testing.assert_array_equal(result.soft_partial[:, 0], [3.0, 5.0])
    np.testing.assert_array_equal(result.tail_discount[:, 0], [0.0, 0.0])
    np.testing.assert_array_equal(result.bootstrap_tail[:, 0], [0.0, 0.0])
    np.testing.assert_array_equal(result.soft_bootstrapped[:, 0], [3.0, 5.0])
    assert result.censored.all()


def test_nonfinite_values_are_rejected_only_when_they_describe_valid_data():
    with pytest.raises(ValueError, match="rewards must be finite"):
        compute_returns(
            rewards=np.array([[np.nan]]),
            log_probs=np.array([[0.0]]),
            valid=np.array([[True]]),
            terminals=np.array([[False]]),
            truncations=np.array([[False]]),
            next_soft_values=np.array([[1.0]]),
            gamma=0.9,
            alpha=0.2,
        )

    with pytest.raises(ValueError, match="log_probs must be finite"):
        compute_returns(
            rewards=np.array([[1.0]]),
            log_probs=np.array([[np.inf]]),
            valid=np.array([[True]]),
            terminals=np.array([[False]]),
            truncations=np.array([[False]]),
            next_soft_values=np.array([[1.0]]),
            gamma=0.9,
            alpha=0.2,
        )

    with pytest.raises(ValueError, match="must not be infinite"):
        compute_returns(
            rewards=np.array([[1.0]]),
            log_probs=np.array([[0.0]]),
            valid=np.array([[True]]),
            terminals=np.array([[False]]),
            truncations=np.array([[True]]),
            next_soft_values=np.array([[np.inf]]),
            gamma=0.9,
            alpha=0.2,
        )
