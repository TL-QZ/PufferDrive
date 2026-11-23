import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.development.collect_binned_probes import (
    _assigned_bin,
    _bin_label,
    _resolve_shard_path,
    _source_row_for_state,
    _worker_assignment,
)


def test_worker_assignments_are_disjoint_and_cover_the_selection():
    states = [{"state_id": f"state-{index}"} for index in range(11)]

    assignments = [_worker_assignment(states, worker_index, 4) for worker_index in range(4)]

    flattened_ids = [state["state_id"] for assignment in assignments for state in assignment]
    assert len(flattened_ids) == len(set(flattened_ids))
    assert set(flattened_ids) == {state["state_id"] for state in states}
    with pytest.raises(ValueError, match="worker_index"):
        _worker_assignment(states, 4, 4)


def test_structured_assigned_bin_is_preserved_and_grouped_by_bin_id():
    assigned_bin = {
        "axis": "nearest_visible_goal_distance_m",
        "bin_index": 3,
        "bin_id": "nearest_visible_goal_distance_m:03",
        "lower_m": 20.0,
        "upper_m": 40.0,
    }

    assert _assigned_bin({"assigned_bin": assigned_bin}) == assigned_bin
    assert _bin_label(assigned_bin) == "nearest_visible_goal_distance_m:03"


def test_source_row_checks_selected_identity_fingerprints_and_geometry():
    state = {
        "state_id": "Town01/episode_00000/frame_000004/agent_12_slot_3",
        "frame_idx": 4,
        "slot": 3,
        "agent_id": 12,
        "current_goal_idx": 1,
        "goal_count": 3,
        "fingerprint": "native-frame",
        "observation_fingerprint": "observation-frame",
        "distances_m": {
            "nearest_observed_partner_center_distance_m": 4.5,
            "nearest_observed_road_boundary_distance_m": None,
            "nearest_visible_goal_distance_m": 25.0,
            "current_goal_distance_m": 26.0,
        },
    }
    arrays = {
        "frame_idx": np.asarray([4]),
        "slot": np.asarray([3]),
        "agent_id": np.asarray([12]),
        "current_goal_idx": np.asarray([1]),
        "goal_count": np.asarray([3]),
        "preaction_eligible": np.zeros((8, 6), dtype=np.bool_),
        "joint_actions": np.zeros((8, 6), dtype=np.int32),
        "fingerprints": np.asarray(["", "", "", "", "native-frame", "", "", ""]),
        "observation_fingerprints": np.asarray(["", "", "", "", "observation-frame", "", "", ""]),
        "nearest_observed_partner_center_distance_m": np.asarray([4.5], dtype=np.float32),
        "nearest_observed_road_boundary_distance_m": np.asarray([np.nan], dtype=np.float32),
        "nearest_visible_goal_distance_m": np.asarray([25.0], dtype=np.float32),
        "current_goal_distance_m": np.asarray([26.0], dtype=np.float32),
    }
    arrays["preaction_eligible"][4, 3] = True

    source_row = _source_row_for_state(state, {"frame_count": 8, "agent_count": 6}, arrays)

    assert source_row["agent_id"] == 12
    assert source_row["fingerprint"] == "native-frame"
    assert source_row["distances_m"]["nearest_observed_road_boundary_distance_m"] is None


def test_source_shard_cannot_escape_records_directory(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()

    with pytest.raises(ValueError, match="escapes"):
        _resolve_shard_path(records_dir, "../../outside.npz")
