import json

import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.development.collect_binned_probes import (
    _assigned_bin,
    _bin_label,
    _resolve_shard_path,
    _source_row_for_state,
    _update_collection_summary,
    _worker_assignment,
    build_argument_parser,
    validate_args,
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

    state["distances_m"]["nearest_visible_goal_distance_m"] = 19.0
    state["distance_corrections_m"] = {
        "nearest_visible_goal_distance_m": 19.0,
        "source_nearest_visible_goal_distance_m": 25.0,
        "current_goal_idx": 1,
        "resolved_goal_idx": 2,
        "fulfilled_goal_count": 1,
        "goal_radius_m": 12.0,
        "goal_distance_m": 19.0,
    }
    corrected_source_row = _source_row_for_state(
        state, {"frame_count": 8, "agent_count": 6}, arrays
    )
    assert corrected_source_row["distances_m"]["nearest_visible_goal_distance_m"] == 25.0

    state["distance_corrections_m"]["source_nearest_visible_goal_distance_m"] = 23.0
    with pytest.raises(ValueError, match="source value"):
        _source_row_for_state(state, {"frame_count": 8, "agent_count": 6}, arrays)


def test_source_shard_cannot_escape_records_directory(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()

    with pytest.raises(ValueError, match="escapes"):
        _resolve_shard_path(records_dir, "../../outside.npz")


def test_round_four_summary_counts_all_repeats(tmp_path):
    records_dir = tmp_path / "records"
    probes_dir = records_dir / "probes"
    probes_dir.mkdir(parents=True)
    selection = {
        "states": [{
            "state_id": "Town01/episode_00000/frame_000008/agent_0001_slot_0001",
            "assigned_bin": {"bin_id": "nearest_agent:00"},
        }],
    }
    probe_metadata = {
        "status": "complete",
        "run_signature": "round-four-signature",
        "state_id": selection["states"][0]["state_id"],
        "assigned_bin": selection["states"][0]["assigned_bin"],
        "branches": [
            {"metrics_mean": {"episode_return": float(action_idx)}}
            for action_idx in range(48)
        ],
    }
    (probes_dir / "probe.json").write_text(json.dumps(probe_metadata))

    summary = _update_collection_summary(
        records_dir, selection, "round-four-signature", repeats=4,
        record_horizons=(8, 32, 128, 1280, 2560),
    )

    assert summary["format_version"] == 2
    assert summary["repeats"] == 4
    assert summary["expected_branch_count"] == 48
    assert summary["completed_branch_count"] == 48
    assert summary["record_horizons"] == [8, 32, 128, 1280, 2560]
    assert summary["complete"] is True


def test_round_four_cli_accepts_repeats_and_ordered_horizons():
    args = build_argument_parser().parse_args([
        "--model-dir", "runs/fourth_20261124/pql",
        "--source-model-dir", "runs/second_20261122/pql",
        "--worker-index", "0", "--workers", "24",
        "--repeats", "4", "--record-horizons", "8", "32", "128", "1280", "2560",
    ])

    validate_args(args)

    assert args.repeats == 4
    assert args.record_horizons == [8, 32, 128, 1280, 2560]
