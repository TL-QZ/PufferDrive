import numpy as np
import pytest
import torch

from pufferlib.sac_returns import compute_pql_targets
from project.soft_actor_critic_q_calibration.probe_returns import (
    compute_probe_measurements,
)


def test_probe_returns_exclude_forced_action_entropy_and_report_soft_tail():
    rewards = np.array([[1.0], [2.0], [3.0]])
    log_probs = np.array([[np.nan], [-0.5], [-1.0]])
    valid = np.ones((3, 1), dtype=bool)
    terminals = np.array([[False], [False], [False]])
    truncations = np.array([[False], [False], [True]])
    next_soft_values = np.array([[10.0], [20.0], [4.0]])

    result = compute_probe_measurements(
        rewards,
        log_probs,
        valid,
        terminals,
        truncations,
        next_soft_values,
        gamma=0.5,
        alpha=0.2,
        return_estimator="n_step",
    )

    # The forced t=0 action has no entropy term; future sampled actions do.
    expected_reward_return = 1.0 + 0.5 * 2.0 + 0.25 * 3.0
    expected_soft_return = expected_reward_return + 0.5 * 0.2 * 0.5 + 0.25 * 0.2
    assert result["discounted_reward_return"].dtype == np.float64
    assert result["discounted_reward_return"].tolist() == [expected_reward_return]
    assert result["standard_soft_return"].tolist() == [expected_soft_return]
    assert result["terminal"].tolist() == [False]
    assert result["truncated"].tolist() == [True]
    assert result["censored"].tolist() == [True]
    assert result["observed_horizon"].tolist() == [3]
    assert result["return_tail_discount"].tolist() == [0.125]
    assert result["critic_bootstrapped_soft_tail"].tolist() == [0.5]
    assert result["critic_bootstrapped_soft_return"].tolist() == [expected_soft_return + 0.5]
    assert result["n_step_1_target"].tolist() == [6.0]
    assert result["actual_critic_target"].tolist() == [6.0]


def test_probe_pql_target_matches_repository_recurrence_and_horizon_cap():
    rewards = np.array(
        [[1.0, -2.0, 3.0], [0.5, 3.0, 4.0], [2.0, 4.0, 5.0], [9.0, 5.0, 6.0]],
        dtype=np.float64,
    )
    log_probs = np.zeros_like(rewards)
    valid = np.array(
        [[True, True, True], [True, True, True], [True, False, False], [False, False, False]]
    )
    terminals = np.array(
        [[False, False, False], [False, False, True], [True, False, False], [False, False, False]]
    )
    truncations = np.zeros_like(valid)
    next_soft_values = np.array(
        [[0.25, 1.0, -0.5], [0.5, 2.0, np.nan], [np.nan, 3.0, 4.0], [4.0, np.nan, np.nan]]
    )
    pql_horizon = 2
    gamma = 0.9
    trace_lambda = 0.7

    result = compute_probe_measurements(
        rewards,
        log_probs,
        valid,
        terminals,
        truncations,
        next_soft_values,
        gamma=gamma,
        alpha=0.0,
        return_estimator="pql",
        pql_horizon=pql_horizon,
        pql_lambda=trace_lambda,
    )

    # The repository target uses float32 raw rewards, gamma discounts and soft
    # next-state values, with one independent packed segment per branch.
    packed_rewards = torch.tensor([1.0, 0.5, -2.0, 3.0, 3.0, 4.0], dtype=torch.float32)
    packed_discounts = torch.tensor([gamma, gamma, gamma, gamma, gamma, 0.0], dtype=torch.float32)
    packed_next_values = torch.tensor([0.25, 0.5, 1.0, 2.0, -0.5, 0.0], dtype=torch.float32)
    expected = compute_pql_targets(
        packed_rewards,
        packed_discounts,
        packed_next_values,
        torch.tensor([0, 2, 4, 6], dtype=torch.int64),
        trace_lambda,
    ).numpy()

    np.testing.assert_array_equal(result["pql_target"], expected[[0, 2, 4]].astype(np.float64))
    np.testing.assert_array_equal(result["actual_critic_target"], result["pql_target"])
    assert result["pql_horizon_used"].tolist() == [2, 2, 2]
    assert result["pql_horizon_capped"].tolist() == [True, False, False]
    assert result["pql_terminal"].tolist() == [False, False, True]
    assert result["pql_censored"].tolist() == [True, True, False]
    assert result["pql_tail_discount"].tolist() == [gamma**2, gamma**2, 0.0]
    # The first branch's terminal is beyond the PQL cap, so this target retains
    # a bootstrap at the cap.  The third branch includes its terminal in PQL.
    assert result["terminal"].tolist() == [True, False, True]
    assert result["return_tail_discount"].tolist() == [0.0, gamma**2, 0.0]


def test_terminal_takes_precedence_over_truncation_and_missing_tail():
    result = compute_probe_measurements(
        rewards=np.array([[2.0]]),
        log_probs=np.array([[np.nan]]),
        valid=np.array([[True]]),
        terminals=np.array([[True]]),
        truncations=np.array([[True]]),
        next_soft_values=np.array([[np.nan]]),
        gamma=0.9,
        alpha=0.2,
        return_estimator="pql",
    )

    assert result["terminal"].tolist() == [True]
    assert result["truncated"].tolist() == [False]
    assert result["return_tail_discount"].tolist() == [0.0]
    assert result["critic_bootstrapped_soft_tail"].tolist() == [0.0]
    assert result["critic_bootstrapped_soft_return"].tolist() == [2.0]
    assert result["n_step_1_target"].tolist() == [2.0]
    assert result["pql_target"].tolist() == [2.0]


def test_missing_nonterminal_bootstrap_stays_nan_in_tail_and_targets():
    result = compute_probe_measurements(
        rewards=np.array([[1.0], [2.0]]),
        log_probs=np.array([[np.nan], [-0.5]]),
        valid=np.ones((2, 1), dtype=bool),
        terminals=np.zeros((2, 1), dtype=bool),
        truncations=np.array([[False], [True]]),
        next_soft_values=np.array([[3.0], [np.nan]]),
        gamma=0.5,
        alpha=0.2,
        return_estimator="pql",
        pql_lambda=1.0,
    )

    assert np.isnan(result["critic_bootstrapped_soft_tail"][0])
    assert np.isnan(result["critic_bootstrapped_soft_return"][0])
    assert np.isnan(result["pql_target"][0])
    assert result["n_step_1_target"].tolist() == [2.5]


@pytest.mark.parametrize(
    ("valid", "terminals", "truncations", "message"),
    [
        (
            np.array([[True], [False], [True]]),
            np.zeros((3, 1), dtype=bool),
            np.zeros((3, 1), dtype=bool),
            "valid must be a prefix",
        ),
        (
            np.array([[True], [True], [False]]),
            np.array([[True], [False], [False]]),
            np.zeros((3, 1), dtype=bool),
            "only on the final valid transition",
        ),
        (
            np.array([[True], [False]]),
            np.array([[False], [True]]),
            np.zeros((2, 1), dtype=bool),
            "false on invalid padding",
        ),
    ],
)
def test_probe_rejects_invalid_prefixes_and_boundary_flags(valid, terminals, truncations, message):
    with pytest.raises(ValueError, match=message):
        compute_probe_measurements(
            rewards=np.ones(valid.shape),
            log_probs=np.zeros(valid.shape),
            valid=valid,
            terminals=terminals,
            truncations=truncations,
            next_soft_values=np.zeros(valid.shape),
            gamma=0.9,
            alpha=0.2,
            return_estimator="n_step",
        )
