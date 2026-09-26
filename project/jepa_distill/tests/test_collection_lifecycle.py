from __future__ import annotations

from pathlib import Path

import pytest

from project.jepa_distill.collection_lifecycle import (
    durable_replace_checkpoint,
    expected_round_manifest_path,
    remove_completed_collection,
    validate_active_collection,
    validate_round_manifest_path,
)


def test_checkpoint_flushes_file_and_directory_before_cleanup(tmp_path, monkeypatch):
    import os
    import stat

    events = []
    original_fsync, original_replace = os.fsync, os.replace

    def fsync(descriptor):
        events.append('directory' if stat.S_ISDIR(os.fstat(descriptor).st_mode) else 'file')
        original_fsync(descriptor)

    def replace(source, target):
        events.append('replace')
        original_replace(source, target)

    monkeypatch.setattr(os, 'fsync', fsync)
    monkeypatch.setattr(os, 'replace', replace)
    temporary_path, checkpoint_path = tmp_path / 'temporary.pt', tmp_path / 'checkpoint.pt'
    temporary_path.write_bytes(b'new checkpoint')
    checkpoint_path.write_bytes(b'previous checkpoint')
    durable_replace_checkpoint(temporary_path, checkpoint_path)
    assert events == ['file', 'replace', 'directory']
    assert checkpoint_path.read_bytes() == b'new checkpoint'


def test_active_collection_allows_only_the_checkpoint_round(tmp_path: Path) -> None:
    training_root = tmp_path / "dataset" / "train"
    active_round = training_root / "round_0002"
    active_round.mkdir(parents=True)

    validate_active_collection(training_root, 2)
    with pytest.raises(FileExistsError, match="stale or extra"):
        validate_active_collection(training_root, None)

    (training_root / "round_0001").mkdir()
    with pytest.raises(FileExistsError, match="stale or extra"):
        validate_active_collection(training_root, 2)


def test_manifest_must_be_at_its_expected_canonical_round_path(tmp_path: Path) -> None:
    training_root = tmp_path / "dataset" / "train"
    expected_path = expected_round_manifest_path(training_root, 0)
    expected_path.parent.mkdir(parents=True)
    expected_path.write_text("{}", encoding="utf-8")

    assert validate_round_manifest_path(training_root, 0, expected_path) == expected_path
    outside_path = tmp_path / "outside" / "manifest.json"
    outside_path.parent.mkdir()
    outside_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="expected round location"):
        validate_round_manifest_path(training_root, 0, outside_path)


def test_cleanup_is_scoped_idempotent_and_preserves_neighbor_data(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    training_root = dataset_root / "train"
    round_path = training_root / "round_0000"
    round_path.mkdir(parents=True)
    (round_path / "large.npy").write_bytes(b"training data")
    validation_path = dataset_root / "validation" / "round_0000"
    validation_path.mkdir(parents=True)
    (validation_path / "manifest.json").write_text("{}", encoding="utf-8")

    assert remove_completed_collection(training_root, 0) is True
    assert remove_completed_collection(training_root, 0) is False
    assert not round_path.exists()
    assert (validation_path / "manifest.json").is_file()


def test_cleanup_refuses_round_symlinks_and_symlinks_inside_round(tmp_path: Path) -> None:
    training_root = tmp_path / "dataset" / "train"
    training_root.mkdir(parents=True)
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    try:
        (training_root / "round_0000").symlink_to(outside_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        validate_active_collection(training_root, 0)
    with pytest.raises(ValueError, match="symlink"):
        remove_completed_collection(training_root, 0)
    (training_root / "round_0000").unlink()

    round_path = training_root / "round_0001"
    round_path.mkdir()
    (outside_root / "keep.txt").write_text("outside", encoding="utf-8")
    (round_path / "linked-outside").symlink_to(outside_root, target_is_directory=True)
    with pytest.raises(ValueError, match="containing a symlink"):
        remove_completed_collection(training_root, 1)
    assert (outside_root / "keep.txt").is_file()
