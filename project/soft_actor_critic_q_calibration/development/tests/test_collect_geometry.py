"""Focused checks for deterministic geometry shard collection and resume."""

import json

import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.development.collect_geometry import (
    GEOMETRY_FIELDS,
    _assemble_episode_arrays,
    _completed_map_frames,
    _enumerate_maps,
    _open_or_create_geometry_manifest,
    _save_episode,
)


def test_cli_requires_model_paths_steps_and_device():
    from project.soft_actor_critic_q_calibration.development.collect_geometry import build_argument_parser

    args = build_argument_parser().parse_args([
        "--model-dir", "project/soft_actor_critic_q_calibration/runs/geometry/pql",
        "--source-model-dir", "project/soft_actor_critic_q_calibration/runs/first_20261114/pql",
        "--steps-per-map", "2560",
        "--device", "cuda:0",
    ])
    assert args.steps_per_map == 2560
    assert args.seed == 20261122
    assert args.map_index is None


def test_explicit_map_enumeration_is_sorted_and_capped_by_training_count(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    for name in ("Town10HD", "Town03", "Town01", "Town05", "Town06", "Town02", "Town07", "Town04", "zzExtra"):
        (map_dir / f"opendrive__{name}.bin").write_bytes(name.encode())

    maps = _enumerate_maps({"map_dir": str(map_dir), "num_maps": 8}, tmp_path)
    assert [row["map_name"] for row in maps] == [
        "Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"
    ]
    assert [row["map_index"] for row in maps] == list(range(8))
    assert all(len(row["sha256"]) == 64 for row in maps)


def test_episode_arrays_contain_one_row_for_each_eligible_frame_slot():
    eligibility = np.array([[True, False, True], [False, True, True]])
    agent_ids = np.array([[10, 11, 12], [10, 11, 12]], dtype=np.int32)
    geometry = {name: np.arange(6, dtype=np.float32).reshape(2, 3) for name in GEOMETRY_FIELDS}
    arrays = _assemble_episode_arrays(
        agent_ids=agent_ids,
        eligibility=eligibility,
        current_goal_indices=np.zeros((2, 3), dtype=np.int32),
        goal_counts=np.ones((2, 3), dtype=np.int32),
        geometry_series=geometry,
        joint_actions=np.zeros((2, 3), dtype=np.int32),
        fingerprints=["a" * 64, "b" * 64],
        observation_fingerprints=["c" * 64, "d" * 64],
    )

    np.testing.assert_array_equal(arrays["frame_idx"], [0, 0, 1, 1])
    np.testing.assert_array_equal(arrays["slot"], [0, 2, 1, 2])
    np.testing.assert_array_equal(arrays["agent_id"], [10, 12, 11, 12])
    np.testing.assert_array_equal(arrays["preaction_eligible"], eligibility)
    assert arrays["joint_actions"].shape == (2, 3)
    assert arrays["fingerprints"].tolist() == ["a" * 64, "b" * 64]
    for name in GEOMETRY_FIELDS:
        assert arrays[name].shape == (4,)


def test_geometry_manifest_is_stable_and_rejects_changed_inputs(tmp_path):
    manifest_path = tmp_path / "geometry_manifest.json"
    settings = {"seed": 12, "steps_per_map": 2560, "maps": ["Town01"]}
    first = _open_or_create_geometry_manifest(manifest_path, settings)
    second = _open_or_create_geometry_manifest(manifest_path, settings)
    assert first["signature"] == second["signature"]
    with pytest.raises(ValueError, match="does not match"):
        _open_or_create_geometry_manifest(manifest_path, {**settings, "seed": 13})


def test_completed_episode_shards_resume_and_detect_corruption(tmp_path):
    references_dir = tmp_path / "references"
    map_record = {
        "map_index": 0,
        "map_name": "Town01",
        "path": "/maps/Town01.bin",
        "sha256": "map-hash",
    }
    arrays = _assemble_episode_arrays(
        agent_ids=np.array([[7, 8]], dtype=np.int32),
        eligibility=np.array([[True, False]]),
        current_goal_indices=np.array([[0, 0]], dtype=np.int32),
        goal_counts=np.array([[1, 0]], dtype=np.int32),
        geometry_series={name: np.array([[1.0, np.nan]], dtype=np.float32) for name in GEOMETRY_FIELDS},
        joint_actions=np.array([[2, 3]], dtype=np.int32),
        fingerprints=["a" * 64],
        observation_fingerprints=["b" * 64],
    )
    episode = {
        "arrays": arrays,
        "frame_count": 1,
        "agent_count": 2,
        "episode_seed": 901,
        "boundary_captured": True,
        "frame_limit_reached": False,
    }
    _save_episode(
        references_dir, map_record, 0, 0, "signature", 1001, 0, 901,
        np.zeros((12, 2), dtype=np.float32), episode,
    )
    completed, frame_count = _completed_map_frames(references_dir, map_record, "signature")
    assert len(completed) == 1
    assert frame_count == 1
    assert json.loads((references_dir / "Town01" / "episode_00000.json").read_text())["episode_seed"] == 901

    shard_path = references_dir / "Town01" / "episode_00000.npz"
    shard_path.write_bytes(shard_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        _completed_map_frames(references_dir, map_record, "signature")
