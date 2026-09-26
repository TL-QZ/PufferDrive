"""Streaming dataset and resumable batching checks for the observation probe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from project.jepa_distill.observation_probe.data import (
    ProbeCollectionDataset,
    iter_collection_batches,
    preflight_probe_data,
    selected_indices,
)


_ARRAY_LAYOUTS = {
    "observations": (np.dtype("float32"), lambda steps, slots, dim, classes: (steps + 1, slots, dim)),
    "controls": (np.dtype("float32"), lambda steps, slots, dim, classes: (steps, slots, 2)),
    "logits": (np.dtype("float32"), lambda steps, slots, dim, classes: (steps, slots, classes)),
    "transition_valid": (np.dtype("bool"), lambda steps, slots, dim, classes: (steps, slots)),
    "eligibility_mask": (np.dtype("bool"), lambda steps, slots, dim, classes: (steps, slots)),
    "terminated": (np.dtype("bool"), lambda steps, slots, dim, classes: (steps, slots)),
    "truncated": (np.dtype("bool"), lambda steps, slots, dim, classes: (steps, slots)),
    "endpoint_valid": (np.dtype("bool"), lambda steps, slots, dim, classes: (steps + 1, slots)),
    "generation": (np.dtype("int64"), lambda steps, slots, dim, classes: (steps + 1, slots)),
}
_ACTION_TABLE = [[0.0, 0.0], [1.0, 1.0]]


def _write_collection(
    directory: Path,
    *,
    split: str = "train",
    round_idx: int = 0,
    base: float = 0.0,
    window_indices: tuple[int, ...] = (4, 1),
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    steps, slots, dimension, classes, horizon = 8, 2, 3, 2, 2
    arrays: dict[str, np.ndarray] = {}
    descriptors: dict[str, dict] = {}
    for name, (dtype, shape_fn) in _ARRAY_LAYOUTS.items():
        shape = shape_fn(steps, slots, dimension, classes)
        array = np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        array[...] = 0
        arrays[name] = array
        descriptors[name] = {"dtype": dtype.name, "path": f"{name}.npy", "shape": list(shape)}

    for timestep in range(steps + 1):
        for slot in range(slots):
            arrays["observations"][timestep, slot] = [
                base + timestep * 10 + slot + feature * 0.1 for feature in range(dimension)
            ]
    for timestep in range(steps):
        arrays["controls"][timestep, :, 0] = base + timestep
        arrays["controls"][timestep, :, 1] = base + timestep + np.arange(slots) * 0.01
    arrays["transition_valid"][:] = True
    arrays["eligibility_mask"][:] = True
    arrays["endpoint_valid"][:] = True
    arrays["generation"][:] = 0
    for array in arrays.values():
        array.flush()
        del array

    window_path = directory / "window_indices.npy"
    np.save(window_path, np.asarray(window_indices, dtype=np.int64))
    manifest = {
        "schema_version": 1,
        "dataset_format": "condition_b_streaming_npy_v1",
        "complete": True,
        "collection_round_idx": round_idx,
        "split": split,
        "simulator_steps": steps,
        "slots": slots,
        "observation_dim": dimension,
        "chunk_length": horizon,
        "num_action_classes": classes,
        "valid_window_count": len(window_indices),
        "observation_layout": {"observation_dim": dimension},
        "action_layout": {
            "control_order": ["longitudinal", "lateral"],
            "normalized_controls": _ACTION_TABLE,
        },
        "effective_config": {},
        "arrays": descriptors,
        "window_indices": {
            "dtype": "int64",
            "path": window_path.name,
            "shape": [len(window_indices)],
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _make_shards(tmp_path: Path) -> tuple[Path, Path]:
    shard_zero = _write_collection(
        tmp_path / "rank_000" / "train" / "round_0000",
        base=0.0,
        window_indices=(4, 1),
    )
    shard_one = _write_collection(
        tmp_path / "rank_001" / "train" / "round_0000",
        base=1000.0,
        window_indices=(6, 3),
    )
    return shard_zero, shard_one


def test_existing_preflight_still_checks_and_pools_train_manifests(tmp_path):
    _make_shards(tmp_path)

    report = preflight_probe_data(
        {
            "data": {
                "collection_root": str(tmp_path),
                "collection_rounds": [0],
                "source_ranks": [0, 1],
            }
        }
    )

    assert report["selected_rounds"] == [0]
    assert report["per_round"][0]["pooled_valid_window_count"] == 4
    assert report["array_audit"].startswith("NPY headers")


def test_dataset_preserves_shard_and_requested_batch_order_and_uses_endpoint_t_plus_k(tmp_path):
    manifest_zero, manifest_one = _make_shards(tmp_path)

    with ProbeCollectionDataset([manifest_zero, manifest_one]) as dataset:
        batch = dataset.get_batch([2, 0, 3, 1])

    expected = [
        (manifest_one, 6, 1000.0),
        (manifest_zero, 4, 0.0),
        (manifest_one, 3, 1000.0),
        (manifest_zero, 1, 0.0),
    ]
    assert len(batch.source_ids) == len(expected)
    for row, (manifest_path, flattened_start, base) in enumerate(expected):
        transition_start, slot = divmod(flattened_start, 2)
        np.testing.assert_allclose(
            batch.current_observations[row].numpy(),
            [base + transition_start * 10 + slot + feature * 0.1 for feature in range(3)],
        )
        np.testing.assert_allclose(
            batch.future_observations[row].numpy(),
            [base + (transition_start + 2) * 10 + slot + feature * 0.1 for feature in range(3)],
        )
        np.testing.assert_allclose(
            batch.executed_controls[row].numpy(),
            [
                [base + timestep, base + timestep + slot * 0.01]
                for timestep in range(transition_start, transition_start + 2)
            ],
        )
        assert batch.source_ids[row].manifest_path == str(manifest_path.resolve())
        assert batch.source_ids[row].window_index == flattened_start


def test_dataset_opens_only_probe_inputs_and_window_validity_arrays(tmp_path, monkeypatch):
    manifest_zero, _manifest_one = _make_shards(tmp_path)
    original_load = np.load
    opened_names: list[str] = []

    def track_load(path, *args, **kwargs):
        opened_names.append(Path(path).name)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", track_load)
    dataset = ProbeCollectionDataset([manifest_zero])
    try:
        dataset.get_batch([0])
    finally:
        dataset.close()

    assert set(opened_names) == {
        "window_indices.npy",
        "observations.npy",
        "controls.npy",
        "transition_valid.npy",
        "endpoint_valid.npy",
        "generation.npy",
    }


@pytest.mark.parametrize(
    ("window_indices", "corruption", "message"),
    [
        ((-1,), None, "outside"),
        ((14,), None, "truncated"),
        ((0,), "endpoint", "invalid observation endpoint"),
        ((0,), "transition", "invalid transition"),
        ((0,), "generation", "generation boundary"),
    ],
)
def test_dataset_rejects_corrupt_selected_window_indices_and_validity(
    tmp_path, window_indices, corruption, message
):
    manifest_path = _write_collection(
        tmp_path / "rank_000" / "train" / "round_0000",
        window_indices=window_indices,
    )
    if corruption == "endpoint":
        array = np.load(manifest_path.parent / "endpoint_valid.npy", mmap_mode="r+")
        array[2, 0] = False
        array.flush()
    elif corruption == "transition":
        array = np.load(manifest_path.parent / "transition_valid.npy", mmap_mode="r+")
        array[0, 0] = False
        array.flush()
    elif corruption == "generation":
        array = np.load(manifest_path.parent / "generation.npy", mmap_mode="r+")
        array[2, 0] = 1
        array.flush()

    dataset = ProbeCollectionDataset([manifest_path])
    try:
        with pytest.raises(ValueError, match=message):
            dataset.get_batch([0])
    finally:
        dataset.close()


@pytest.mark.parametrize("split", ["validation", "test"])
def test_dataset_audits_held_out_splits_explicitly(tmp_path, split):
    manifest_path = _write_collection(
        tmp_path / "rank_000" / split / "round_0000",
        split=split,
        window_indices=(0,),
    )

    with ProbeCollectionDataset([manifest_path], expected_split=split) as dataset:
        assert len(dataset) == 1
        assert dataset.get_batch([0]).source_ids[0].manifest_path == str(manifest_path.resolve())

    with pytest.raises(ValueError, match="expected 'train'"):
        ProbeCollectionDataset([manifest_path])


def test_selected_indices_are_deterministic_capped_and_resume_exactly(tmp_path):
    manifest_zero, manifest_one = _make_shards(tmp_path)
    first_order = selected_indices(19, seed=17, epoch_index=2, max_windows=7, collection_index=3)
    repeated_order = selected_indices(19, seed=17, epoch_index=2, max_windows=7, collection_index=3)
    full_order = selected_indices(19, seed=17, epoch_index=2, collection_index=3)
    next_epoch_order = selected_indices(19, seed=17, epoch_index=3, max_windows=7, collection_index=3)

    assert list(first_order) == list(repeated_order)
    assert list(first_order) == list(full_order[:7])
    assert len(set(first_order)) == 7
    assert list(first_order) != list(next_epoch_order)

    full_epoch = list(
        iter_collection_batches(
            [manifest_zero, manifest_one],
            batch_size=3,
            seed=29,
            epoch_index=4,
            collection_index=1,
        )
    )
    resumed_epoch = list(
        iter_collection_batches(
            [manifest_zero, manifest_one],
            batch_size=3,
            seed=29,
            epoch_index=4,
            start_batch_index=1,
            collection_index=1,
        )
    )
    assert len(full_epoch) == 2
    assert [source for batch in resumed_epoch for source in batch.source_ids] == [
        source for batch in full_epoch[1:] for source in batch.source_ids
    ]


def test_dataset_rejects_zero_window_training_round_and_context_closes(tmp_path):
    empty_manifest = _write_collection(
        tmp_path / "rank_000" / "train" / "round_0000",
        window_indices=(),
    )
    with pytest.raises(ValueError, match="no valid windows"):
        ProbeCollectionDataset([empty_manifest])

    manifest = _write_collection(
        tmp_path / "rank_001" / "train" / "round_0001",
        round_idx=1,
        window_indices=(0,),
    )
    dataset = ProbeCollectionDataset([manifest])
    dataset.close()
    dataset.close()
    with pytest.raises(RuntimeError, match="is closed"):
        dataset.get_batch([0])
