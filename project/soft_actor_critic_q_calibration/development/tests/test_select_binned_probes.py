"""Focused checks for deterministic, unique second-run state selection."""

import json

import numpy as np

from project.soft_actor_critic_q_calibration.development.select_binned_probes import (
    GEOMETRY_FIELDS,
    PRIMARY_AXES,
    _axis_bin_index,
    _load_candidate_pool,
    _select_assignments,
    _sha256_file,
    build_argument_parser,
    build_selection,
)


def _synthetic_pool(row_count=1200):
    map_indices = (np.arange(row_count) % 8).astype(np.int32)
    episode_indices = (np.arange(row_count) // 64 % 4).astype(np.int32)
    rows = {
        "map_index": map_indices,
        "episode_index": episode_indices,
        "frame_idx": (np.arange(row_count) % 256).astype(np.int32),
        "agent_id": (np.arange(row_count) // 256).astype(np.int32),
        "slot": (np.arange(row_count) // 256).astype(np.int32),
        "current_goal_idx": np.zeros(row_count, dtype=np.int32),
        "goal_count": np.ones(row_count, dtype=np.int32),
        "source_shard_index": (map_indices * 4 + episode_indices).astype(np.int32),
    }
    distances = {}
    bin_indices = {}
    for axis_index, (axis, edges) in enumerate(PRIMARY_AXES.items()):
        requested_bin = (np.arange(row_count) + axis_index * 7) % (len(edges) - 1)
        values = np.empty(row_count, dtype=np.float32)
        for bin_index in range(len(edges) - 1):
            selected = requested_bin == bin_index
            lower = edges[bin_index]
            upper = edges[bin_index + 1]
            values[selected] = lower + 0.5 if np.isinf(upper) else (lower + upper) / 2
        distances[axis] = values
        bin_indices[axis] = requested_bin.astype(np.int16)
    distances["current_goal_distance_m"] = np.arange(row_count, dtype=np.float32)
    rows.update(distances)
    map_names = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
    shards = []
    for map_index, map_name in enumerate(map_names):
        for episode_index in range(4):
            shards.append({
                "map_name": map_name,
                "map_path": f"/maps/{map_name}.bin",
                "episode_index": episode_index,
                "layout_seed": 11,
                "layout_draw_index": 0,
                "scene_seed": 12,
                "episode_seed": 13,
                "agent_count": 4,
                "source_shard": f"references/{map_name}/episode_{episode_index:05d}.npz",
                "source_metadata": f"references/{map_name}/episode_{episode_index:05d}.json",
                "fingerprints": np.array(["f" * 64] * 256),
                "observation_fingerprints": np.array(["o" * 64] * 256),
            })
    return {
        "rows": rows,
        "bin_indices": bin_indices,
        "shards": shards,
        "source_shards": [
            {
                "source_shard": shard["source_shard"],
                "source_metadata": shard["source_metadata"],
                "data_sha256": "d" * 64,
            }
            for shard in shards
        ],
        "source_geometry_manifest_signature": "geometry-signature",
    }


def test_axis_bins_are_lower_inclusive_and_missing_values_are_unassigned():
    edges = PRIMARY_AXES["nearest_observed_partner_center_distance_m"]
    values = np.array([0, 4.999, 5, 10, 80, 300, np.nan], dtype=np.float64)

    np.testing.assert_array_equal(_axis_bin_index(values, edges), [0, 0, 1, 2, 5, 5, -1])


def test_selection_is_seeded_unique_and_preserves_all_tags_and_goal_distance():
    pool = _synthetic_pool()
    selection = build_selection(
        pool,
        model="pql",
        source_records_dir="runs/second_20261122/pql/records",
        seed=314,
        per_bin=1,
    )
    repeated = build_selection(
        pool,
        model="pql",
        source_records_dir="runs/second_20261122/pql/records",
        seed=314,
        per_bin=1,
    )

    assert selection["states"] == repeated["states"]
    assert selection["selected_state_count"] == 19
    assert len({state["state_id"] for state in selection["states"]}) == 19
    assert all(len(state["bin_tags"]) == 3 for state in selection["states"])
    assert all(state["assigned_bin"]["bin_id"] in {
        tag["bin_id"] for tag in state["bin_tags"].values()
    } for state in selection["states"])
    assert all("current_goal_distance_m" in state["distances_m"] for state in selection["states"])
    assert all(
        summary["selected_count"] == 1
        for axis_bins in selection["bin_summary"].values()
        for summary in axis_bins
    )
    assert not selection["shortages"]


def test_selector_avoids_adjacent_frames_of_same_car_when_support_allows():
    rows = {
        "map_index": np.zeros(3, dtype=np.int32),
        "episode_index": np.zeros(3, dtype=np.int32),
        "frame_idx": np.array([10, 11, 20], dtype=np.int32),
        "agent_id": np.ones(3, dtype=np.int32),
        "slot": np.zeros(3, dtype=np.int32),
    }
    pool = {
        "rows": rows,
        "bin_indices": {
            axis: np.array([0, 0, 0], dtype=np.int16)
            if axis == next(iter(PRIMARY_AXES)) else np.full(3, -1, dtype=np.int16)
            for axis in PRIMARY_AXES
        },
    }

    assignments, _, relaxed_count = _select_assignments(pool, seed=8, per_bin=2)
    selected_frames = sorted(int(rows["frame_idx"][row["row_index"]]) for row in assignments)

    assert len(assignments) == 2
    assert len({row["row_index"] for row in assignments}) == 2
    assert abs(selected_frames[0] - selected_frames[1]) > 1
    assert relaxed_count == 0


def test_raw_reference_loader_keeps_recorded_slots_and_fingerprints(tmp_path):
    records_dir = tmp_path / "runs" / "second" / "pql" / "records"
    episode_dir = records_dir / "references" / "Town01"
    episode_dir.mkdir(parents=True)
    (records_dir / "geometry_manifest.json").write_text(json.dumps({"signature": "source-signature"}))
    data_path = episode_dir / "episode_00000.npz"
    arrays = {
        "frame_idx": np.array([0, 0], dtype=np.int32),
        "agent_id": np.array([7, 8], dtype=np.int32),
        "slot": np.array([1, 3], dtype=np.int32),
        "current_goal_idx": np.array([0, 2], dtype=np.int32),
        "goal_count": np.array([2, 3], dtype=np.int32),
        "eligible": np.ones(2, dtype=np.bool_),
        "fingerprints": np.array(["f" * 64]),
        "observation_fingerprints": np.array(["o" * 64]),
        "nearest_observed_partner_center_distance_m": np.array([1.0, 6.0], dtype=np.float32),
        "nearest_observed_road_boundary_distance_m": np.array([1.0, 2.0], dtype=np.float32),
        "nearest_visible_goal_distance_m": np.array([5.0, 12.0], dtype=np.float32),
        "current_goal_distance_m": np.array([5.0, np.nan], dtype=np.float32),
    }
    np.savez_compressed(data_path, **arrays)
    metadata = {
        "run_signature": "source-signature",
        "map_index": 0,
        "map_name": "Town01",
        "map_path": "/maps/Town01.bin",
        "episode_index": 0,
        "layout_seed": 10,
        "layout_draw_index": 0,
        "scene_seed": 11,
        "episode_seed": 12,
        "agent_count": 4,
        "frame_count": 1,
        "data_file": data_path.name,
        "data_sha256": _sha256_file(data_path),
    }
    (episode_dir / "episode_00000.json").write_text(json.dumps(metadata))

    pool = _load_candidate_pool(records_dir)

    np.testing.assert_array_equal(pool["rows"]["agent_id"], [7, 8])
    np.testing.assert_array_equal(pool["rows"]["slot"], [1, 3])
    np.testing.assert_array_equal(pool["rows"]["current_goal_idx"], [0, 2])
    assert np.isnan(pool["rows"]["current_goal_distance_m"][1])
    assert pool["shards"][0]["fingerprints"].tolist() == ["f" * 64]


def test_selector_cli_defaults_match_agreed_bin_quota():
    args = build_argument_parser().parse_args([
        "--source-model-dir", "runs/second_20261122/pql",
        "--model-dir", "runs/third_20261122/pql",
    ])
    assert args.seed == 20261122
    assert args.per_bin == 15
    assert len(GEOMETRY_FIELDS) == 4
