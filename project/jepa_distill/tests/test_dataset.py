"""Focused tests for the Condition B shard and window contract."""

import json

import numpy as np
import pytest

from project.jepa_distill.dataset import TrajectoryDataset, WindowReference, build_window_index, validate_manifest


def _write_fixture(tmp_path, *, include_nan=False):
    observations = np.arange(14, dtype=np.float32).reshape(7, 2)
    if include_nan:
        observations[0, 0] = np.nan
    controls = np.ones((5, 2), dtype=np.float32)
    logits = np.arange(15, dtype=np.float32).reshape(5, 3)
    trajectory_offsets = np.array([0, 4, 5], dtype=np.int64)
    observation_offsets = np.array([0, 5, 7], dtype=np.int64)
    valid_transition = np.array([True, True, True, True, False], dtype=bool)
    state_valid = np.array([True, True, True, True, True, True, False], dtype=bool)
    terminated = np.array([False, False, False, False, True], dtype=bool)
    truncated = np.array([False, False, False, False, False], dtype=bool)
    eligibility_mask = np.ones(5, dtype=bool)
    shard_path = tmp_path / "shard.npz"
    np.savez_compressed(
        shard_path,
        observations=observations,
        executed_controls=controls,
        teacher_logits=logits,
        trajectory_offsets=trajectory_offsets,
        observation_offsets=observation_offsets,
        valid_transition=valid_transition,
        state_valid=state_valid,
        terminated=terminated,
        truncated=truncated,
        eligibility_mask=eligibility_mask,
    )
    manifest = {
        "schema_version": 1,
        "chunk_length": 2,
        "num_action_classes": 3,
        "observation_layout": {"observation_dim": 2},
        "shards": [{"path": "shard.npz", "trajectory_count": 2, "transition_count": 5}],
        "splits": {
            "train": {"trajectory_refs": [{"shard_idx": 0, "trajectory_idx": 0}], "trajectory_count": 1},
            "validation": {"trajectory_refs": [], "trajectory_count": 0},
            "test": {"trajectory_refs": [{"shard_idx": 0, "trajectory_idx": 1}], "trajectory_count": 1},
        },
        "trajectory_count": 2,
        "transition_count": 5,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_window_index_and_dataset_preserve_alignment(tmp_path):
    manifest = _write_fixture(tmp_path)
    manifest["_dataset_root"] = str(tmp_path)
    assert build_window_index(manifest, split="train") == (
        WindowReference(0, 0, 0),
        WindowReference(0, 0, 1),
        WindowReference(0, 0, 2),
    )
    dataset = TrajectoryDataset(tmp_path, split="train")
    assert len(dataset) == 3
    sample = dataset[1]
    assert tuple(sample.observations.shape) == (3, 2)
    assert tuple(sample.executed_controls.shape) == (2, 2)
    assert tuple(sample.teacher_logits.shape) == (2, 3)
    np.testing.assert_array_equal(sample.observations.numpy(), np.arange(2, 8, dtype=np.float32).reshape(3, 2))


def test_terminal_endpoint_and_split_boundary_are_rejected(tmp_path):
    manifest = _write_fixture(tmp_path)
    manifest["_dataset_root"] = str(tmp_path)
    # The test trajectory has only one transition and its terminal marker is
    # excluded; it cannot borrow observations from train trajectory 0.
    assert build_window_index(manifest, split="test") == ()
    assert len(TrajectoryDataset(tmp_path, split="test")) == 0


def test_manifest_and_shard_nan_are_rejected(tmp_path):
    manifest = _write_fixture(tmp_path, include_nan=True)
    manifest["_dataset_root"] = str(tmp_path)
    validate_manifest(manifest)
    with pytest.raises(ValueError, match="NaN/Inf"):
        build_window_index(manifest, split="train")
    manifest["observation_layout"]["normalization"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_manifest(manifest)


def test_missing_explicit_split_is_rejected(tmp_path):
    manifest = _write_fixture(tmp_path)
    del manifest["splits"]["test"]
    with pytest.raises(ValueError, match="missing required split"):
        validate_manifest(manifest)
