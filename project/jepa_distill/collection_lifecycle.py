"""Safe ownership and cleanup for one active training collection."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path


def durable_replace_checkpoint(temporary_path: Path, checkpoint_path: Path) -> None:
    """Persist the replacement checkpoint before callers may delete its old data."""
    with open(temporary_path, 'rb') as checkpoint_file:
        os.fsync(checkpoint_file.fileno())
    os.replace(temporary_path, checkpoint_path)
    directory_fd = os.open(Path(checkpoint_path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _round_name(round_idx: int) -> str:
    if isinstance(round_idx, bool) or not isinstance(round_idx, int) or round_idx < 0:
        raise ValueError("collection round index must be a non-negative integer")
    return f"round_{round_idx:04d}"


def _training_root_path(training_root: Path) -> Path:
    root = Path(training_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = Path(os.path.abspath(root))
    resolved_root = root.resolve(strict=False)
    if root != resolved_root:
        raise ValueError(f"training collection root traverses a symlink: {root}")
    return root


def expected_round_manifest_path(training_root: Path, round_idx: int) -> Path:
    """Return the sole canonical manifest location owned by this round."""
    return _training_root_path(training_root) / _round_name(round_idx) / "manifest.json"


def validate_round_manifest_path(
    training_root: Path, round_idx: int, manifest_path: Path | str
) -> Path:
    """Require a collected or resumed manifest at its canonical round path."""
    expected_path = expected_round_manifest_path(training_root, round_idx)
    candidate_path = Path(manifest_path).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = Path.cwd() / candidate_path
    candidate_path = Path(os.path.abspath(candidate_path))
    if candidate_path != expected_path:
        raise ValueError(
            "collection manifest path does not match its expected round location: "
            f"expected {expected_path}, got {candidate_path}"
        )
    round_path = expected_path.parent
    if round_path.is_symlink() or expected_path.is_symlink():
        raise ValueError(f"collection round contains a symlink: {round_path}")
    if not expected_path.is_file():
        raise FileNotFoundError(f"collection manifest does not exist: {expected_path}")
    return expected_path


def validate_active_collection(
    training_root: Path, active_round_idx: int | None
) -> None:
    """Reject stale, extra, or unsafe round paths under a training root.

    A root may be absent before the first collection. At most one direct child
    named ``round_NNNN`` is permitted, and it must be the checkpoint's active
    round when one is supplied.
    """
    root = _training_root_path(training_root)
    expected_name = _round_name(active_round_idx) if active_round_idx is not None else None
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"training collection root is not a directory: {root}")

    round_names: list[str] = []
    for child in root.iterdir():
        if not child.name.startswith("round_"):
            continue
        if child.is_symlink():
            raise ValueError(f"training collection round is a symlink: {child}")
        if not child.is_dir():
            raise ValueError(f"unexpected non-directory round path: {child}")
        round_names.append(child.name)

    unexpected_names = [name for name in round_names if name != expected_name]
    if unexpected_names:
        raise FileExistsError(
            f"training collection root contains stale or extra rounds: "
            f"{', '.join(sorted(unexpected_names))}; expected only {expected_name!r}"
        )
def remove_completed_collection(training_root: Path, round_idx: int) -> bool:
    """Remove exactly one completed round; return False when already absent."""
    root = _training_root_path(training_root)
    round_path = root / _round_name(round_idx)
    try:
        round_stat = round_path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(round_stat.st_mode):
        raise ValueError(f"refusing to remove a symlink collection round: {round_path}")
    if not stat.S_ISDIR(round_stat.st_mode):
        raise ValueError(f"refusing to remove a non-directory collection round: {round_path}")
    if round_path.resolve(strict=True).parent != root.resolve(strict=True):
        raise ValueError(f"collection round resolves outside its training root: {round_path}")

    for child in round_path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"refusing to remove a collection containing a symlink: {child}")
    shutil.rmtree(round_path)
    return True
