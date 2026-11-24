import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.development.collect_ppo_probes import (
    PPO_SEGMENT_ROWS,
    _native_ppo_advantages,
    _reward_for_ppo,
    prepare_ppo_segment,
)


def _manual_gae(values, rewards, dones, gamma, gae_lambda):
    advantages = np.zeros_like(values, dtype=np.float32)
    accumulator = np.float32(0.0)
    for timestep in range(len(values) - 2, -1, -1):
        next_nonterminal = np.float32(1.0) - np.float32(dones[timestep + 1])
        delta = (
            np.float32(rewards[timestep + 1])
            + np.float32(gamma) * np.float32(values[timestep + 1]) * next_nonterminal
            - np.float32(values[timestep])
        )
        accumulator = delta + (
            np.float32(gamma) * np.float32(gae_lambda) * accumulator * next_nonterminal
        )
        advantages[timestep] = accumulator
    return advantages


def test_full_segment_matches_training_row_alignment_and_bootstrap():
    gamma = 0.9
    gae_lambda = 0.5
    values = np.linspace(-1.0, 1.0, PPO_SEGMENT_ROWS, dtype=np.float32)
    transition_rewards = np.linspace(0.25, 2.0, PPO_SEGMENT_ROWS - 1, dtype=np.float32)
    target = prepare_ppo_segment(
        values,
        transition_rewards,
        np.zeros(PPO_SEGMENT_ROWS - 1, dtype=np.bool_),
        np.zeros(PPO_SEGMENT_ROWS - 1, dtype=np.bool_),
        np.ones(PPO_SEGMENT_ROWS - 1, dtype=np.bool_),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    expected_rewards = np.concatenate(([0.0], transition_rewards)).astype(np.float32)
    expected_dones = np.zeros(PPO_SEGMENT_ROWS, dtype=np.float32)
    expected_advantages = _manual_gae(values, expected_rewards, expected_dones, gamma, gae_lambda)
    assert target["ppo_segment_rows"] == 128
    assert target["ppo_segment_transitions"] == 127
    assert target["ppo_segment_boundary"] == "segment_limit"
    assert target["ppo_slot_valid"].all()
    assert target["ppo_value_valid"].all()
    np.testing.assert_array_equal(target["ppo_training_rewards"], expected_rewards)
    np.testing.assert_array_equal(target["ppo_training_dones"], expected_dones)
    np.testing.assert_allclose(target["ppo_advantages"], expected_advantages, rtol=0.0, atol=1e-6)
    assert target["ppo_value_target"] == pytest.approx(float(values[0] + expected_advantages[0]))


def test_early_terminal_stops_target_without_post_terminal_rows():
    target = prepare_ppo_segment(
        np.asarray([10.0, 99.0, 98.0], dtype=np.float32),
        np.asarray([4.0, 5.0], dtype=np.float32),
        np.asarray([True, False]),
        np.asarray([False, False]),
        np.asarray([True, True]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert target["ppo_segment_rows"] == 2
    assert target["ppo_segment_transitions"] == 1
    assert target["ppo_segment_boundary"] == "terminal"
    assert target["ppo_slot_valid"][:2].all()
    assert not target["ppo_slot_valid"][2:].any()
    assert target["ppo_training_dones"][1] == 1.0
    assert target["ppo_values"][1] == 0.0
    assert not target["ppo_value_valid"][1]
    assert target["ppo_value_target"] == pytest.approx(4.0)


def test_truncation_uses_previous_state_value_bootstrap_proxy():
    target = prepare_ppo_segment(
        np.asarray([10.0, 5.0], dtype=np.float32),
        np.asarray([4.0], dtype=np.float32),
        np.asarray([False]),
        np.asarray([True]),
        np.asarray([True]),
        gamma=0.9,
        gae_lambda=0.95,
        use_value_bootstrapping=True,
    )

    assert target["ppo_segment_boundary"] == "truncation"
    assert target["ppo_training_rewards"][1] == pytest.approx(13.0)
    assert target["ppo_training_dones"][1] == 1.0
    assert target["ppo_value_target"] == pytest.approx(13.0)


def test_learner_mask_gap_is_a_boundary_and_remains_distinct():
    target = prepare_ppo_segment(
        np.asarray([2.0, 99.0, 98.0], dtype=np.float32),
        np.asarray([1.0, 3.0], dtype=np.float32),
        np.asarray([False, False]),
        np.asarray([False, False]),
        np.asarray([False, True]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert target["ppo_segment_rows"] == 2
    assert target["ppo_segment_boundary"] == "learner_mask_gap"
    assert target["ppo_training_dones"][1] == 1.0
    assert not target["ppo_training_masks"][1]
    assert target["ppo_value_target"] == pytest.approx(1.0)


def test_normalized_reward_matches_puffer_sign_log1p_transform():
    rewards = np.asarray([-3.0, 0.0, 4.0], dtype=np.float32)

    normalized = _reward_for_ppo(rewards, True)

    np.testing.assert_allclose(
        normalized,
        np.sign(rewards) * np.log1p(np.abs(rewards)),
        rtol=0.0,
        atol=1e-7,
    )


def test_native_advantage_helper_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="shared length"):
        _native_ppo_advantages([1.0, 2.0], [0.0], [0.0, 0.0], [1.0, 1.0], 0.9, 0.95)
