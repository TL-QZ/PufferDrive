import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from run_probes import (
    build_argument_parser,
    build_branch_tasks,
    build_probe_selection,
    chunked,
    continuation_seed,
    open_or_create_manifest,
    observation_fingerprint,
    probe_is_complete,
    probe_paths,
    sanitize_json,
    save_probe,
    sample_actions,
    select_probe_egos,
    select_probe_frames,
)


def test_probe_frames_follow_eligible_span_and_are_distinct():
    eligible = np.zeros((101, 6), dtype=np.bool_)
    eligible[10:91, :2] = True
    selected = select_probe_frames(eligible)

    assert selected["eligible_span"] == {"first_frame": 10, "last_frame": 90}
    assert selected["selected_count"] == 3
    assert selected["shortage_count"] == 0
    assert len({row["frame_idx"] for row in selected["frames"]}) == 3
    assert [row["target_fraction"] for row in selected["frames"]] == [0.1, 0.3, 0.5]
    # Targets map to the first half of the observed eligible span, without
    # accessing Q values, outcomes, rewards, or agent features.
    assert [row["frame_idx"] for row in selected["frames"]] == [18, 34, 50]


def test_probe_frame_selection_reports_missing_eligible_support():
    eligible = np.zeros((20, 4), dtype=np.bool_)
    eligible[3, :2] = True
    eligible[12, 1:3] = True
    selected = select_probe_frames(eligible)

    assert selected["selected_count"] == 2
    assert selected["shortage_count"] == 1
    assert [row["frame_idx"] for row in selected["frames"]] == [3, 12]

    none = select_probe_frames(np.zeros((10, 3), dtype=np.bool_))
    assert none["frames"] == []
    assert none["shortage_count"] == 3


def test_ego_selection_is_seeded_and_limited_to_eligible_agents():
    candidates = [0, 2, 5, 8]
    selected = select_probe_egos(candidates, seed=91, episode_index=1, frame_idx=20)
    assert selected == select_probe_egos(candidates, seed=91, episode_index=1, frame_idx=20)
    assert len(selected) == 2
    assert set(selected) <= set(candidates)
    assert selected == sorted(selected)
    assert select_probe_egos([4], seed=91, episode_index=1, frame_idx=20) == []


def test_probe_selection_tracks_ego_shortage():
    eligible = np.zeros((8, 4), dtype=np.bool_)
    eligible[2, :2] = True
    eligible[5, 1:3] = True
    selection = build_probe_selection(eligible, seed=7, episode_index=0)
    assert selection["selected_probe_count"] == 4
    assert selection["shortage_count"] == 2
    assert all(probe["ego_idx"] in (0, 1, 2) for probe in selection["probes"])


def test_branch_seeds_are_paired_and_independent_of_batch_chunking():
    tasks = build_branch_tasks("ep0/frame12/ego3", 17, 0, 12, 3, repeats=8)
    assert len(tasks) == 96
    assert all(tasks[repeat * 12 + action]["action_idx"] == action for repeat in range(8) for action in range(12))
    for repeat in range(8):
        paired = {tasks[repeat * 12 + action]["continuation_seed"] for action in range(12)}
        assert len(paired) == 1
    assert len({tasks[repeat * 12]["continuation_seed"] for repeat in range(8)}) == 8
    assert continuation_seed(17, 0, 12, 3, 0) == tasks[0]["continuation_seed"]

    by_id = {task["task_id"]: task["continuation_seed"] for task in tasks}
    for batch_size in (1, 7, 24, 97):
        chunked_ids = [task["task_id"] for batch in chunked(tasks, batch_size) for task in batch]
        assert chunked_ids == [task["task_id"] for task in tasks]
        assert {task_id: by_id[task_id] for task_id in chunked_ids} == by_id


def test_categorical_action_sampling_uses_explicit_rng():
    probabilities = np.array([[0.0, 1.0], [0.25, 0.75], [1.0, 0.0]])
    first = sample_actions(probabilities, np.random.default_rng(11))
    second = sample_actions(probabilities, np.random.default_rng(11))
    np.testing.assert_array_equal(first, second)
    assert first[0] == 1
    assert first[2] == 0

    with pytest.raises(ValueError, match="sum to one"):
        sample_actions(np.array([[0.2, 0.2]]), np.random.default_rng(0))


def test_json_sanitization_keeps_nonfinite_values_as_null():
    clean = sanitize_json({"nan": float("nan"), "inf": np.float32(np.inf), "array": np.array([1.0, np.nan])})
    assert clean == {"nan": None, "inf": None, "array": [1.0, None]}
    json.dumps(clean, allow_nan=False)


def test_manifest_resume_requires_same_immutable_inputs(tmp_path):
    manifest = {
        "checkpoint": {"path": "/tmp/checkpoint.pt", "sha256": "abc"},
        "resolved_config": {"gamma": 0.999},
        "source_hashes": {"runner": "def"},
        "map_hashes": {"scene.bin": "map-hash"},
        "native_build_fingerprint": {"private_binding_sha256": "native-hash"},
        "run_settings": {"episodes": 2, "repeats": 8},
    }
    first = open_or_create_manifest(tmp_path, manifest)
    second = open_or_create_manifest(tmp_path, manifest)
    assert first["signature"] == second["signature"]

    changed = dict(manifest)
    changed["run_settings"] = {"episodes": 3, "repeats": 8}
    with pytest.raises(ValueError, match="does not match"):
        open_or_create_manifest(tmp_path, changed)

    changed = dict(manifest)
    changed["map_hashes"] = {"scene.bin": "changed-map-hash"}
    with pytest.raises(ValueError, match="does not match"):
        open_or_create_manifest(tmp_path, changed)


def test_probe_bundle_is_atomic_resumable_and_records_signature(tmp_path):
    probe_id = "episode_000/frame_000012/ego_003"
    signature = "run-signature"
    assert not probe_is_complete(tmp_path, probe_id, signature)
    save_probe(tmp_path, probe_id, signature, {
                   "probe": {"ego_idx": 3}, "branch_task_count": 1,
                   "repeats": 1, "action_count": 1, "branch_task_ids": ["task0"],
               },
               {"rewards": np.array([[1.0]]), "valid": np.array([[True]])})
    assert probe_is_complete(
        tmp_path, probe_id, signature, expected_branch_count=1,
        expected_repeats=1, expected_action_count=1, expected_task_ids=["task0"],
    )
    assert not probe_is_complete(tmp_path, probe_id, "another-signature")
    metadata_path, data_path = probe_paths(tmp_path, probe_id)
    assert json.loads(metadata_path.read_text())["probe"]["ego_idx"] == 3
    with np.load(data_path) as data:
        np.testing.assert_array_equal(data["rewards"], [[1.0]])
    data_path.write_bytes(data_path.read_bytes() + b"corrupt")
    assert not probe_is_complete(tmp_path, probe_id, signature)


def test_observation_fingerprint_is_stable_and_sensitive_to_obs_values():
    observations = np.array([[0.0, 1.0], [-0.5, 0.25]], dtype=np.float32)
    first = observation_fingerprint(observations)
    assert first == observation_fingerprint(observations.copy())
    changed = observations.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], 1.0)
    assert first != observation_fingerprint(changed)


def test_cli_defaults_and_validation_gates(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test")
    args = build_argument_parser().parse_args(["--checkpoint", str(checkpoint), "--output", str(tmp_path / "out")])
    assert args.episodes == 2
    assert args.repeats == 8
    assert args.branch_batch_size == 24
    assert args.cpu_threads == 2
    assert args.reference_cap is None
    assert args.max_probes is None
