"""Geometry summaries decoded from the actual Drive observation layout."""

import math

import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.development.initial_geometry import initial_geometry


@pytest.fixture
def observation_and_config():
    config = {
        "num_goals": 2,
        "reward_conditioning": False,
        "obs_slots_partners_n": 2,
        "obs_slots_lane_n": 2,
        "obs_slots_boundary_n": 1,
        "obs_slots_traffic_controls_n": 1,
        "obs_dropout_lane": 0.5,
        "obs_dropout_boundary": 0.0,
        "obs_norm_goal_offset_m": 100.0,
        "obs_norm_xy_offset_m": 100.0,
        "obs_norm_z_m": 10.0,
        "obs_norm_road_seg_length_m": 5.0,
    }
    # Native layout: ego, goals, partners, lanes, boundaries, traffic, valid counts.
    observation = np.zeros(10 + 6 + 18 + 9 + 9 + 7 + 4, dtype=np.float32)
    observation[10:13] = [0.3, 0.4, 0.0]
    observation[13:16] = [0.6, 0.8, 0.0]

    partner_start = 16
    observation[partner_start:partner_start + 3] = [0.3, 0.4, 0.0]
    observation[partner_start + 9:partner_start + 12] = [0.6, 0.8, 0.0]

    lane_start = partner_start + 18
    observation[lane_start:lane_start + 7] = [0.1, 0.0, 0.0, 2.0, 0.2, 1.0, 0.0]

    boundary_start = lane_start + 9
    observation[boundary_start:boundary_start + 7] = [0.0, 0.2, 0.0, 1.0, 0.2, 1.0, 0.0]

    observation[-4:] = [1.0, 1.0, 2.0, 0.0]
    return observation, config


def test_geometry_decodes_counts_normalization_and_finite_segment_distance(observation_and_config):
    observation, config = observation_and_config
    result = initial_geometry(observation, config, current_goal_idx=0, goal_count=2)
    assert result["nearest_observed_partner_center_distance_m"] == pytest.approx(50.0)
    assert result["nearest_observed_lane_segment_distance_m"] == pytest.approx(0.0, abs=1e-6)
    assert result["nearest_observed_road_boundary_distance_m"] == pytest.approx(20.0)
    assert result["nearest_visible_goal_distance_m"] == pytest.approx(50.0)
    assert result["current_goal_distance_m"] == pytest.approx(50.0)


def test_goal_slot_zero_is_valid_and_completed_goals_are_excluded(observation_and_config):
    observation, config = observation_and_config
    observation[10:13] = [0.9, 0.9, 0.0]  # already completed goal slot
    observation[13:16] = [0.0, 0.0, 0.0]  # valid next goal at ego
    result = initial_geometry(observation, config, current_goal_idx=1, goal_count=2)
    assert result["nearest_visible_goal_distance_m"] == 0.0
    assert result["current_goal_distance_m"] == 0.0


def test_missing_observed_slots_are_none_and_empty_goal_tail_is_none(observation_and_config):
    observation, config = observation_and_config
    observation[-4:] = [0.0, 0.0, 0.0, 0.0]
    result = initial_geometry(observation, config, current_goal_idx=2, goal_count=2)
    assert result["nearest_observed_partner_center_distance_m"] is None
    assert result["nearest_observed_lane_segment_distance_m"] is None
    assert result["nearest_observed_road_boundary_distance_m"] is None
    assert result["nearest_visible_goal_distance_m"] is None
    assert result["current_goal_distance_m"] is None


def test_invalid_observation_size_and_valid_counts_fail_fast(observation_and_config):
    observation, config = observation_and_config
    with pytest.raises(ValueError, match="observation size"):
        initial_geometry(observation[:-1], config, current_goal_idx=0, goal_count=2)
    observation[-4] = 3.0
    with pytest.raises(ValueError, match="valid count"):
        initial_geometry(observation, config, current_goal_idx=0, goal_count=2)
