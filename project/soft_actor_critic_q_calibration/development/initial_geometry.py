"""Recover visible initial geometry from a Drive policy observation."""

from collections.abc import Mapping
import math

import numpy as np

from pufferlib.ocean.drive import binding


PARTNER_FEATURE_COUNT = int(binding.PARTNER_FEATURES)
LANE_FEATURE_COUNT = int(binding.LANE_FEATURES)
BOUNDARY_FEATURE_COUNT = int(binding.BOUNDARY_FEATURES)
GOAL_FEATURE_COUNT = int(binding.GOAL_FEATURES)
OBS_VALID_COUNT_FEATURE_COUNT = int(binding.OBS_VALID_COUNT_FEATURES)


def initial_geometry(observation, config, *, current_goal_idx, goal_count):
    """Return metric distances derivable from the probing ego's observation.

    Road distances use the finite horizontal segment represented by each observed
    road feature. Partner and goal distances use horizontal relative coordinates.
    Missing slots are
    identified by native valid counts or goal metadata, never by zero values.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping of Drive environment settings")
    observation = np.asarray(observation, dtype=np.float64)
    if observation.ndim != 1 or not np.all(np.isfinite(observation)):
        raise ValueError("observation must be a finite one-dimensional array")

    num_goals = _positive_or_zero_int(config, "num_goals")
    partner_slots = _positive_or_zero_int(config, "obs_slots_partners_n")
    lane_slots = _road_slot_count(config, "lane")
    boundary_slots = _road_slot_count(config, "boundary")
    traffic_slots = _positive_or_zero_int(config, "obs_slots_traffic_controls_n")
    reward_coef_count = int(binding.NUM_REWARD_COEFS) if config.get("reward_conditioning", False) else 0

    ego_features = int(binding.EGO_FEATURES)
    goal_start = ego_features + reward_coef_count
    partner_start = goal_start + num_goals * GOAL_FEATURE_COUNT
    lane_start = partner_start + partner_slots * PARTNER_FEATURE_COUNT
    boundary_start = lane_start + lane_slots * LANE_FEATURE_COUNT
    traffic_start = boundary_start + boundary_slots * BOUNDARY_FEATURE_COUNT
    counts_start = traffic_start + traffic_slots * int(binding.TRAFFIC_CONTROL_FEATURES)
    expected_size = counts_start + OBS_VALID_COUNT_FEATURE_COUNT
    if observation.size != expected_size:
        raise ValueError(f"Expected Drive observation size {expected_size}, got {observation.size}")

    lane_count, boundary_count, partner_count, _ = _valid_counts(
        observation[counts_start:], (lane_slots, boundary_slots, partner_slots, traffic_slots)
    )
    xy_norm = _positive_finite(config, "obs_norm_xy_offset_m")
    goal_norm = _positive_finite(config, "obs_norm_goal_offset_m")
    segment_norm = _positive_finite(config, "obs_norm_road_seg_length_m")

    partner_distances = []
    for slot in range(partner_count):
        start = partner_start + slot * PARTNER_FEATURE_COUNT
        x = observation[start] * xy_norm
        y = observation[start + 1] * xy_norm
        partner_distances.append(math.hypot(x, y))

    lane_distances = _road_segment_distances(
        observation, lane_start, lane_count, LANE_FEATURE_COUNT, xy_norm, segment_norm
    )
    boundary_distances = _road_segment_distances(
        observation, boundary_start, boundary_count, BOUNDARY_FEATURE_COUNT, xy_norm, segment_norm
    )

    first_goal = _nonnegative_int(current_goal_idx, "current_goal_idx")
    total_goal_count = _nonnegative_int(goal_count, "goal_count")
    if first_goal > total_goal_count:
        raise ValueError("current_goal_idx cannot exceed goal_count")
    valid_goal_end = min(total_goal_count, num_goals)
    goal_distances = []
    for goal_idx in range(first_goal, valid_goal_end):
        start = goal_start + goal_idx * GOAL_FEATURE_COUNT
        x = observation[start] * goal_norm
        y = observation[start + 1] * goal_norm
        goal_distances.append(math.hypot(x, y))

    return {
        "nearest_observed_partner_center_distance_m": min(partner_distances, default=None),
        "nearest_observed_lane_segment_distance_m": min(lane_distances, default=None),
        "nearest_observed_road_boundary_distance_m": min(boundary_distances, default=None),
        "nearest_visible_goal_distance_m": min(goal_distances, default=None),
        "current_goal_distance_m": goal_distances[0] if goal_distances else None,
    }


def _road_segment_distances(observation, start, count, feature_count, xy_norm, segment_norm):
    distances = []
    for slot in range(count):
        feature_start = start + slot * feature_count
        midpoint_x = observation[feature_start] * xy_norm
        midpoint_y = observation[feature_start + 1] * xy_norm
        half_length = observation[feature_start + 3] * segment_norm
        direction_x = observation[feature_start + 5]
        direction_y = observation[feature_start + 6]
        direction_length = math.hypot(direction_x, direction_y)
        if half_length < 0.0:
            raise ValueError(f"Invalid road segment feature at slot {slot}")
        if half_length == 0.0:
            if direction_length != 0.0:
                raise ValueError(f"Invalid zero-length road segment feature at slot {slot}")
            along_segment = 0.0
        else:
            if not math.isclose(direction_length, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError(f"Invalid road segment direction at slot {slot}")
            projection = -(midpoint_x * direction_x + midpoint_y * direction_y)
            along_segment = min(half_length, max(-half_length, projection))
        closest_x = midpoint_x + along_segment * direction_x
        closest_y = midpoint_y + along_segment * direction_y
        distances.append(math.hypot(closest_x, closest_y))
    return distances


def _valid_counts(observation, capacities):
    counts = []
    for value, capacity in zip(observation, capacities):
        rounded = round(float(value))
        if abs(value - rounded) > 1e-4 or rounded < 0 or rounded > capacity:
            raise ValueError(f"Invalid Drive observation valid count {value} for capacity {capacity}")
        counts.append(rounded)
    return counts


def _road_slot_count(config, name):
    kept_name = f"obs_slots_{name}_kept"
    if kept_name in config:
        return _positive_or_zero_int(config, kept_name)
    slots = _positive_or_zero_int(config, f"obs_slots_{name}_n")
    dropout = float(config.get(f"obs_dropout_{name}", 0.0))
    if not math.isfinite(dropout):
        raise ValueError(f"obs_dropout_{name} must be finite")
    from pufferlib.ocean.drive.drive import compute_effective_road_obs_count

    return int(compute_effective_road_obs_count(slots, dropout))


def _positive_or_zero_int(config, name):
    return _nonnegative_int(config.get(name), name)


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_finite(config, name):
    value = float(config[name])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value
