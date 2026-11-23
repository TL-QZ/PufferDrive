"""Vectorized geometry summaries decoded from Drive policy observations."""

from collections.abc import Mapping

import numpy as np

from pufferlib.ocean.drive import binding

from .initial_geometry import _positive_finite, _positive_or_zero_int, _road_slot_count


PARTNER_FEATURE_COUNT = int(binding.PARTNER_FEATURES)
LANE_FEATURE_COUNT = int(binding.LANE_FEATURES)
BOUNDARY_FEATURE_COUNT = int(binding.BOUNDARY_FEATURES)
GOAL_FEATURE_COUNT = int(binding.GOAL_FEATURES)
OBS_VALID_COUNT_FEATURE_COUNT = int(binding.OBS_VALID_COUNT_FEATURES)

_OUTPUT_FIELDS = (
    "nearest_observed_partner_center_distance_m",
    "nearest_observed_road_boundary_distance_m",
    "nearest_visible_goal_distance_m",
    "current_goal_distance_m",
)


def geometry_batch(observations, config, *, current_goal_idx, goal_count):
    """Return four per-observation distances using Drive's native observation layout.

    ``observations`` must have shape ``(batch, observation_size)``. Goal indices
    and counts may be scalars shared by the batch or integer vectors of length
    ``batch``. Missing geometry is represented by ``np.nan``.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping of Drive environment settings")
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2 or not np.all(np.isfinite(observations)):
        raise ValueError("observations must be a finite two-dimensional array")
    batch_size = observations.shape[0]

    current_goal_indices = _batch_nonnegative_ints(current_goal_idx, "current_goal_idx", batch_size)
    goal_counts = _batch_nonnegative_ints(goal_count, "goal_count", batch_size)
    if np.any(current_goal_indices > goal_counts):
        raise ValueError("current_goal_idx cannot exceed goal_count")

    num_goals = _positive_or_zero_int(config, "num_goals")
    partner_slots = _positive_or_zero_int(config, "obs_slots_partners_n")
    lane_slots = _road_slot_count(config, "lane")
    boundary_slots = _road_slot_count(config, "boundary")
    traffic_slots = _positive_or_zero_int(config, "obs_slots_traffic_controls_n")
    reward_coef_count = int(binding.NUM_REWARD_COEFS) if config.get("reward_conditioning", False) else 0

    goal_start = int(binding.EGO_FEATURES) + reward_coef_count
    partner_start = goal_start + num_goals * GOAL_FEATURE_COUNT
    lane_start = partner_start + partner_slots * PARTNER_FEATURE_COUNT
    boundary_start = lane_start + lane_slots * LANE_FEATURE_COUNT
    traffic_start = boundary_start + boundary_slots * BOUNDARY_FEATURE_COUNT
    counts_start = traffic_start + traffic_slots * int(binding.TRAFFIC_CONTROL_FEATURES)
    expected_size = counts_start + OBS_VALID_COUNT_FEATURE_COUNT
    if observations.shape[1] != expected_size:
        raise ValueError(f"Expected Drive observation size {expected_size}, got {observations.shape[1]}")

    valid_counts = _valid_counts_batch(
        observations[:, counts_start:], (lane_slots, boundary_slots, partner_slots, traffic_slots)
    )
    xy_norm = _positive_finite(config, "obs_norm_xy_offset_m")
    goal_norm = _positive_finite(config, "obs_norm_goal_offset_m")
    segment_norm = _positive_finite(config, "obs_norm_road_seg_length_m")

    partner_distance = _nearest_point_distance_batch(
        observations,
        partner_start,
        partner_slots,
        PARTNER_FEATURE_COUNT,
        xy_norm,
        valid_counts[:, 2],
    )
    boundary_distance = _nearest_road_distance_batch(
        observations,
        boundary_start,
        boundary_slots,
        BOUNDARY_FEATURE_COUNT,
        valid_counts[:, 1],
        xy_norm,
        segment_norm,
    )
    goal_distances = _goal_distances_batch(
        observations,
        goal_start,
        num_goals,
        goal_norm,
        current_goal_indices,
        goal_counts,
    )

    if goal_distances.shape[1] == 0:
        nearest_goal_distance = np.full(batch_size, np.nan, dtype=np.float64)
    else:
        nearest_goal_distance = np.min(np.where(np.isfinite(goal_distances), goal_distances, np.inf), axis=1)
        nearest_goal_distance[~np.isfinite(nearest_goal_distance)] = np.nan
    current_distance = np.full(batch_size, np.nan, dtype=np.float64)
    has_current_goal = current_goal_indices < np.minimum(goal_counts, num_goals)
    if np.any(has_current_goal):
        current_distance[has_current_goal] = goal_distances[
            np.flatnonzero(has_current_goal), current_goal_indices[has_current_goal]
        ]

    return dict(zip(_OUTPUT_FIELDS, (partner_distance, boundary_distance, nearest_goal_distance, current_distance)))


def _batch_nonnegative_ints(values, name, batch_size):
    array = np.asarray(values)
    if array.ndim == 0:
        if isinstance(values, (bool, np.bool_)) or not isinstance(values, (int, np.integer)) or values < 0:
            raise ValueError(f"{name} must be a non-negative integer or a vector of them")
        return np.full(batch_size, int(values), dtype=np.int64)
    if array.ndim != 1 or array.shape[0] != batch_size or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be a scalar or an integer vector with length {batch_size}")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return array.astype(np.int64, copy=False)


def _valid_counts_batch(values, capacities):
    rounded = np.rint(values)
    invalid = (np.abs(values - rounded) > 1e-4) | (rounded < 0)
    for column, capacity in enumerate(capacities):
        invalid[:, column] |= rounded[:, column] > capacity
    if np.any(invalid):
        row_idx, column_idx = np.argwhere(invalid)[0]
        raise ValueError(
            f"Invalid Drive observation valid count {values[row_idx, column_idx]} "
            f"for capacity {capacities[column_idx]}"
        )
    return rounded.astype(np.int64)


def _nearest_point_distance_batch(observations, start, slots, feature_count, coordinate_norm, valid_counts):
    batch_size = observations.shape[0]
    if slots == 0:
        return np.full(batch_size, np.nan, dtype=np.float64)
    features = observations[:, start : start + slots * feature_count].reshape(batch_size, slots, feature_count)
    distances = np.hypot(features[:, :, 0] * coordinate_norm, features[:, :, 1] * coordinate_norm)
    valid = np.arange(slots)[None, :] < valid_counts[:, None]
    nearest = np.min(np.where(valid, distances, np.inf), axis=1)
    nearest[~np.isfinite(nearest)] = np.nan
    return nearest


def _nearest_road_distance_batch(
    observations, start, slots, feature_count, valid_counts, coordinate_norm, segment_norm
):
    batch_size = observations.shape[0]
    if slots == 0:
        return np.full(batch_size, np.nan, dtype=np.float64)
    features = observations[:, start : start + slots * feature_count].reshape(batch_size, slots, feature_count)
    valid = np.arange(slots)[None, :] < valid_counts[:, None]

    midpoint_x = features[:, :, 0] * coordinate_norm
    midpoint_y = features[:, :, 1] * coordinate_norm
    half_length = features[:, :, 3] * segment_norm
    direction_x = features[:, :, 5]
    direction_y = features[:, :, 6]
    direction_length = np.hypot(direction_x, direction_y)

    invalid_half_length = valid & (half_length < 0.0)
    if np.any(invalid_half_length):
        _, slot_idx = np.argwhere(invalid_half_length)[0]
        raise ValueError(f"Invalid road segment feature at slot {slot_idx}")
    invalid_zero_direction = valid & (half_length == 0.0) & (direction_length != 0.0)
    if np.any(invalid_zero_direction):
        _, slot_idx = np.argwhere(invalid_zero_direction)[0]
        raise ValueError(f"Invalid zero-length road segment feature at slot {slot_idx}")
    invalid_direction = valid & (half_length > 0.0) & ~np.isclose(
        direction_length, 1.0, rtol=1e-4, atol=1e-4
    )
    if np.any(invalid_direction):
        _, slot_idx = np.argwhere(invalid_direction)[0]
        raise ValueError(f"Invalid road segment direction at slot {slot_idx}")

    projection = -(midpoint_x * direction_x + midpoint_y * direction_y)
    along_segment = np.clip(projection, -half_length, half_length)
    along_segment = np.where(half_length == 0.0, 0.0, along_segment)
    closest_x = midpoint_x + along_segment * direction_x
    closest_y = midpoint_y + along_segment * direction_y
    distances = np.hypot(closest_x, closest_y)
    nearest = np.min(np.where(valid, distances, np.inf), axis=1)
    nearest[~np.isfinite(nearest)] = np.nan
    return nearest


def _goal_distances_batch(
    observations, start, goal_slots, goal_norm, current_goal_indices, goal_counts
):
    batch_size = observations.shape[0]
    if goal_slots == 0:
        return np.empty((batch_size, 0), dtype=np.float64)
    features = observations[:, start : start + goal_slots * GOAL_FEATURE_COUNT].reshape(
        batch_size, goal_slots, GOAL_FEATURE_COUNT
    )
    distances = np.hypot(features[:, :, 0] * goal_norm, features[:, :, 1] * goal_norm)
    goal_indices = np.arange(goal_slots)[None, :]
    visible = (goal_indices >= current_goal_indices[:, None]) & (goal_indices < goal_counts[:, None])
    return np.where(visible, distances, np.nan)
