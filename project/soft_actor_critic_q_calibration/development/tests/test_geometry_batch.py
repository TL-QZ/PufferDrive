"""Batch geometry summaries must match the scalar observation decoder."""

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from project.soft_actor_critic_q_calibration.development.geometry_batch import geometry_batch
from project.soft_actor_critic_q_calibration.development.initial_geometry import initial_geometry


FIELDS = (
    "nearest_observed_partner_center_distance_m",
    "nearest_observed_road_boundary_distance_m",
    "nearest_visible_goal_distance_m",
    "current_goal_distance_m",
)


@pytest.fixture
def observations_and_config():
    config = {
        "num_goals": 3,
        "reward_conditioning": False,
        "obs_slots_partners_n": 2,
        "obs_slots_lane_kept": 1,
        "obs_slots_boundary_kept": 2,
        "obs_slots_traffic_controls_n": 1,
        "obs_norm_goal_offset_m": 100.0,
        "obs_norm_xy_offset_m": 100.0,
        "obs_norm_road_seg_length_m": 5.0,
    }
    observation_size = (
        int(binding.EGO_FEATURES)
        + 3 * int(binding.GOAL_FEATURES)
        + 2 * int(binding.PARTNER_FEATURES)
        + int(binding.LANE_FEATURES)
        + 2 * int(binding.BOUNDARY_FEATURES)
        + int(binding.TRAFFIC_CONTROL_FEATURES)
        + int(binding.OBS_VALID_COUNT_FEATURES)
    )
    observations = np.zeros((4, observation_size), dtype=np.float32)
    goal_start = int(binding.EGO_FEATURES)
    partner_start = goal_start + 3 * int(binding.GOAL_FEATURES)
    lane_start = partner_start + 2 * int(binding.PARTNER_FEATURES)
    boundary_start = lane_start + int(binding.LANE_FEATURES)
    counts_start = boundary_start + 2 * int(binding.BOUNDARY_FEATURES) + int(binding.TRAFFIC_CONTROL_FEATURES)

    observations[:, goal_start : goal_start + 3 * int(binding.GOAL_FEATURES)] = np.array(
        [
            [0.1, 0.2, 0.0, 0.3, 0.4, 0.0, 0.5, 0.6, 0.0],
            [0.8, 0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    observations[0, partner_start : partner_start + 2] = [0.3, 0.4]
    observations[0, partner_start + int(binding.PARTNER_FEATURES) : partner_start + int(binding.PARTNER_FEATURES) + 2] = [0.0, 0.2]
    observations[1, partner_start : partner_start + 2] = [0.5, 0.0]

    # Boundary rows are midpoint x/y, ignored z, half-length, width, unit direction x/y.
    observations[0, boundary_start : boundary_start + 7] = [0.2, 0.0, 0.0, 1.0, 0.2, 1.0, 0.0]
    observations[0, boundary_start + int(binding.BOUNDARY_FEATURES) : boundary_start + int(binding.BOUNDARY_FEATURES) + 7] = [
        0.0, 0.2, 0.0, 1.0, 0.2, 1.0, 0.0
    ]
    observations[1, boundary_start : boundary_start + 7] = [0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]
    observations[3, boundary_start : boundary_start + 7] = [0.5, 0.0, 0.0, 0.2, 0.2, 0.0, 1.0]

    observations[:, counts_start : counts_start + 4] = [
        [0.0, 2.0, 2.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
    ]
    return observations, config


def test_geometry_batch_matches_scalar_for_per_row_goal_metadata(observations_and_config):
    observations, config = observations_and_config
    current_goal_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    goal_counts = np.array([3, 3, 3, 3], dtype=np.int64)

    batch_result = geometry_batch(
        observations,
        config,
        current_goal_idx=current_goal_indices,
        goal_count=goal_counts,
    )
    scalar_results = [
        initial_geometry(
            observation,
            config,
            current_goal_idx=int(current_goal_idx),
            goal_count=int(goal_count),
        )
        for observation, current_goal_idx, goal_count in zip(observations, current_goal_indices, goal_counts)
    ]

    assert tuple(batch_result) == FIELDS
    for field in FIELDS:
        np.testing.assert_allclose(
            batch_result[field],
            [np.nan if result[field] is None else result[field] for result in scalar_results],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        assert batch_result[field].shape == (len(observations),)


def test_geometry_batch_accepts_shared_goal_metadata_and_marks_missing_as_nan(observations_and_config):
    observations, config = observations_and_config
    observations[:, -int(binding.OBS_VALID_COUNT_FEATURES) :] = 0.0
    result = geometry_batch(observations, config, current_goal_idx=3, goal_count=3)
    for field in FIELDS:
        assert result[field].shape == (len(observations),)
        assert np.all(np.isnan(result[field]))


def test_geometry_batch_rejects_invalid_goal_vectors(observations_and_config):
    observations, config = observations_and_config
    with pytest.raises(ValueError, match="integer vector"):
        geometry_batch(observations, config, current_goal_idx=np.array([0.0] * len(observations)), goal_count=3)
    with pytest.raises(ValueError, match="cannot exceed"):
        geometry_batch(observations, config, current_goal_idx=4, goal_count=3)
