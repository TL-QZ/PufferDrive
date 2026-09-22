"""Validated trajectory shards and overlap-aware Condition B windows.

The collector stores each trajectory once in a compressed ``.npz`` shard. The
arrays are flattened across trajectories and paired with offsets so that a
window cannot silently cross a trajectory, reset generation, or shard
boundary. This module deliberately knows nothing about Drive or its C
extension; it only consumes the manifest and NumPy arrays written by
``collect.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]
WindowIndex = Sequence["WindowReference"]
_SCHEMA_VERSION = 1
_REQUIRED_SPLITS = ("train", "validation", "test")


class WindowReference(NamedTuple):
    """Locate a window by shard, trajectory, and transition start."""

    shard_idx: int
    trajectory_idx: int
    start_step_idx: int


class TrainingSample(NamedTuple):
    """One CPU window before collation.

    ``observations`` is ``[K+1,D_o]`` from ``t`` through ``t+K``;
    ``executed_controls`` is ``[K,2]`` in normalized actual-control units; and
    ``teacher_logits`` is ``[K,C]``. ``K`` and ``C`` come from the manifest.
    """

    observations: torch.Tensor
    executed_controls: torch.Tensor
    teacher_logits: torch.Tensor


class TrainingBatch(NamedTuple):
    """Collated independent windows owned by the training loop."""

    observations: torch.Tensor
    executed_controls: torch.Tensor
    teacher_logits: torch.Tensor


def _is_finite_scalar(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _validate_json_values(value: Any, path: str = "manifest") -> None:
    """Reject NaN/Inf and non-JSON values at the metadata boundary."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} has a non-string key {key!r}")
            _validate_json_values(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_values(child, f"{path}[{index}]")
        return
    if isinstance(value, np.ndarray):
        raise ValueError(f"{path} must be JSON metadata, not a NumPy array")
    if isinstance(value, Path):
        raise ValueError(f"{path} must use a string path")
    if not _is_finite_scalar(value):
        raise ValueError(f"{path} contains a non-finite number")


def _as_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a non-negative integer")
    value_int = int(value)
    if value_int < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value_int


def _as_positive_int(value: Any, field: str) -> int:
    value_int = _as_nonnegative_int(value, field)
    if value_int <= 0:
        raise ValueError(f"{field} must be positive")
    return value_int


def _split_mapping(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    splits = manifest.get("splits")
    if splits is None:
        splits = manifest.get("split_metadata")
    if not isinstance(splits, Mapping):
        raise ValueError("manifest.splits must be a mapping")
    return splits


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate manifest metadata before reading any trajectory window.

    Validation is strict at the external-data boundary. It checks schema,
    layout, counts, split identity, and finite metadata; shard contents and
    file existence are checked when the dataset opens them.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    _validate_json_values(manifest)

    schema_version = manifest.get("schema_version")
    if schema_version not in (_SCHEMA_VERSION, str(_SCHEMA_VERSION), "condition_b_dataset_v1"):
        raise ValueError(f"unsupported manifest schema_version {schema_version!r}")

    chunk_length = _as_positive_int(manifest.get("chunk_length"), "manifest.chunk_length")
    num_action_classes = _as_positive_int(
        manifest.get("num_action_classes"), "manifest.num_action_classes"
    )
    observation_layout = manifest.get("observation_layout")
    if not isinstance(observation_layout, Mapping):
        raise ValueError("manifest.observation_layout must be a mapping")
    observation_dim = observation_layout.get("observation_dim", observation_layout.get("dim"))
    observation_dim = _as_positive_int(observation_dim, "manifest.observation_layout.observation_dim")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest.shards must be a non-empty list")
    total_trajectory_count = 0
    total_transition_count = 0
    for shard_idx, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise ValueError(f"manifest.shards[{shard_idx}] must be a mapping")
        shard_path = shard.get("path")
        if not isinstance(shard_path, str) or not shard_path:
            raise ValueError(f"manifest.shards[{shard_idx}].path must be a non-empty string")
        trajectory_count = _as_nonnegative_int(
            shard.get("trajectory_count"), f"manifest.shards[{shard_idx}].trajectory_count"
        )
        transition_count = _as_nonnegative_int(
            shard.get("transition_count"), f"manifest.shards[{shard_idx}].transition_count"
        )
        if trajectory_count == 0 and transition_count != 0:
            raise ValueError(f"manifest.shards[{shard_idx}] has transitions but no trajectories")
        total_trajectory_count += trajectory_count
        total_transition_count += transition_count
        shard_observation_dim = shard.get("observation_dim", observation_dim)
        if _as_positive_int(
            shard_observation_dim, f"manifest.shards[{shard_idx}].observation_dim"
        ) != observation_dim:
            raise ValueError(f"manifest.shards[{shard_idx}] observation width disagrees with layout")
        shard_chunk_length = shard.get("chunk_length", chunk_length)
        if _as_positive_int(shard_chunk_length, f"manifest.shards[{shard_idx}].chunk_length") != chunk_length:
            raise ValueError(f"manifest.shards[{shard_idx}] chunk length disagrees with manifest")
        shard_class_count = shard.get("num_action_classes", num_action_classes)
        if _as_positive_int(
            shard_class_count, f"manifest.shards[{shard_idx}].num_action_classes"
        ) != num_action_classes:
            raise ValueError(f"manifest.shards[{shard_idx}] action class count disagrees with manifest")

    splits = _split_mapping(manifest)
    missing_splits = [split for split in _REQUIRED_SPLITS if split not in splits]
    if missing_splits:
        raise ValueError(f"manifest.splits is missing required split metadata: {missing_splits}")
    seen_refs: set[tuple[int, int]] = set()
    for split_name, split_entry in splits.items():
        if not isinstance(split_name, str) or not split_name:
            raise ValueError("manifest split names must be non-empty strings")
        if not isinstance(split_entry, Mapping):
            raise ValueError(f"manifest.splits[{split_name!r}] must be a mapping")
        refs = split_entry.get("trajectory_refs", split_entry.get("trajectories"))
        if not isinstance(refs, list):
            raise ValueError(f"manifest.splits[{split_name!r}].trajectory_refs must be a list")
        declared_trajectory_count = split_entry.get("trajectory_count", len(refs))
        if _as_nonnegative_int(
            declared_trajectory_count, f"manifest.splits[{split_name!r}].trajectory_count"
        ) != len(refs):
            raise ValueError(f"manifest.splits[{split_name!r}] trajectory_count disagrees with refs")
        for ref_idx, ref in enumerate(refs):
            if not isinstance(ref, Mapping):
                raise ValueError(
                    f"manifest.splits[{split_name!r}].trajectory_refs[{ref_idx}] must be a mapping"
                )
            shard_idx = _as_nonnegative_int(ref.get("shard_idx"), "trajectory ref shard_idx")
            trajectory_idx = _as_nonnegative_int(ref.get("trajectory_idx"), "trajectory ref trajectory_idx")
            if shard_idx >= len(shards):
                raise ValueError(f"trajectory ref shard_idx {shard_idx} is out of range")
            if trajectory_idx >= _as_nonnegative_int(
                shards[shard_idx].get("trajectory_count"), f"manifest.shards[{shard_idx}].trajectory_count"
            ):
                raise ValueError(f"trajectory ref trajectory_idx {trajectory_idx} is out of range")
            reference = (shard_idx, trajectory_idx)
            if reference in seen_refs:
                raise ValueError(f"trajectory {reference} appears in more than one split")
            seen_refs.add(reference)

    declared_trajectory_count = manifest.get("trajectory_count", total_trajectory_count)
    if _as_nonnegative_int(declared_trajectory_count, "manifest.trajectory_count") != total_trajectory_count:
        raise ValueError("manifest.trajectory_count disagrees with shard metadata")
    declared_transition_count = manifest.get("transition_count", total_transition_count)
    if _as_nonnegative_int(declared_transition_count, "manifest.transition_count") != total_transition_count:
        raise ValueError("manifest.transition_count disagrees with shard metadata")

    action_layout = manifest.get("action_layout")
    if action_layout is not None:
        if not isinstance(action_layout, Mapping):
            raise ValueError("manifest.action_layout must be a mapping")
        normalized_table = action_layout.get("normalized_controls")
        if normalized_table is not None:
            table = np.asarray(normalized_table)
            if table.shape != (num_action_classes, 2):
                raise ValueError("manifest.action_layout.normalized_controls must have shape [C,2]")
            if not np.isfinite(table).all():
                raise ValueError("manifest.action_layout.normalized_controls contains NaN/Inf")
            if np.any(table < -1.000001) or np.any(table > 1.000001):
                raise ValueError("manifest.action_layout.normalized_controls must be in [-1,1]")


def _manifest_root(manifest: Mapping[str, Any]) -> Optional[Path]:
    for key in ("_dataset_root", "dataset_root"):
        root = manifest.get(key)
        if root is not None:
            if not isinstance(root, (str, Path)):
                raise ValueError(f"manifest.{key} must be a path string")
            return Path(root).expanduser().resolve()
    return None


def _shard_path(manifest: Mapping[str, Any], shard_idx: int) -> Path:
    shard_path = Path(manifest["shards"][shard_idx]["path"])
    if shard_path.is_absolute():
        return shard_path
    root = _manifest_root(manifest)
    if root is None:
        raise ValueError("relative shard paths require manifest.dataset_root or TrajectoryDataset")
    return (root / shard_path).resolve()


def _require_array(shard: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    value = shard.get(name)
    if value is None:
        raise ValueError(f"shard is missing required array {name!r}")
    if not isinstance(value, np.ndarray):
        raise ValueError(f"shard array {name!r} is not a NumPy array")
    return value


def _validate_shard_arrays(
    manifest: Mapping[str, Any], shard_idx: int, loaded: Mapping[str, np.ndarray]
) -> Mapping[str, np.ndarray]:
    """Validate dtypes, offsets, finite values, and per-trajectory lengths."""

    observation_layout = manifest["observation_layout"]
    observation_dim = int(observation_layout.get("observation_dim", observation_layout.get("dim")))
    num_action_classes = int(manifest["num_action_classes"])
    observations = _require_array(loaded, "observations")
    controls = _require_array(loaded, "executed_controls")
    logits = _require_array(loaded, "teacher_logits")
    trajectory_offsets = _require_array(loaded, "trajectory_offsets")
    observation_offsets = _require_array(loaded, "observation_offsets")
    valid_transition = _require_array(loaded, "valid_transition")
    state_valid = _require_array(loaded, "state_valid")
    terminated = _require_array(loaded, "terminated")
    truncated = _require_array(loaded, "truncated")
    eligibility_mask = _require_array(loaded, "eligibility_mask")

    if observations.dtype != np.float32 or controls.dtype != np.float32 or logits.dtype != np.float32:
        raise ValueError(f"shard {shard_idx} data arrays must use float32")
    if observations.ndim != 2 or observations.shape[1] != observation_dim:
        raise ValueError(f"shard {shard_idx} observations must have shape [N,{observation_dim}]")
    if controls.ndim != 2 or controls.shape[1] != 2:
        raise ValueError(f"shard {shard_idx} executed_controls must have shape [N,2]")
    if logits.ndim != 2 or logits.shape[1] != num_action_classes:
        raise ValueError(f"shard {shard_idx} teacher_logits must have shape [N,{num_action_classes}]")
    for name, array in (("trajectory_offsets", trajectory_offsets), ("observation_offsets", observation_offsets)):
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"shard {shard_idx} {name} must be one-dimensional integers")
    trajectory_count = int(manifest["shards"][shard_idx]["trajectory_count"])
    if trajectory_offsets.shape != (trajectory_count + 1,):
        raise ValueError(f"shard {shard_idx} trajectory_offsets has the wrong length")
    if observation_offsets.shape != (trajectory_count + 1,):
        raise ValueError(f"shard {shard_idx} observation_offsets has the wrong length")
    if trajectory_offsets[0] != 0 or observation_offsets[0] != 0:
        raise ValueError(f"shard {shard_idx} offsets must start at zero")
    if np.any(np.diff(trajectory_offsets) < 0) or np.any(np.diff(observation_offsets) < 0):
        raise ValueError(f"shard {shard_idx} offsets must be monotonic")
    if int(trajectory_offsets[-1]) != controls.shape[0] or int(trajectory_offsets[-1]) != logits.shape[0]:
        raise ValueError(f"shard {shard_idx} transition offsets disagree with data arrays")
    if int(observation_offsets[-1]) != observations.shape[0]:
        raise ValueError(f"shard {shard_idx} observation offsets disagree with observations")
    if int(trajectory_offsets[-1]) != int(manifest["shards"][shard_idx]["transition_count"]):
        raise ValueError(f"shard {shard_idx} transition_count disagrees with arrays")

    transition_count = controls.shape[0]
    for name, array in (
        ("valid_transition", valid_transition),
        ("terminated", terminated),
        ("truncated", truncated),
        ("eligibility_mask", eligibility_mask),
    ):
        if array.shape != (transition_count,) or array.dtype != np.bool_:
            raise ValueError(f"shard {shard_idx} {name} must be bool with one value per transition")
    if state_valid.shape != (observations.shape[0],) or state_valid.dtype != np.bool_:
        raise ValueError(f"shard {shard_idx} state_valid must be bool with one value per observation")
    for name, array in (("observations", observations), ("executed_controls", controls), ("teacher_logits", logits)):
        if not np.isfinite(array).all():
            raise ValueError(f"shard {shard_idx} {name} contains NaN/Inf")

    for trajectory_idx in range(trajectory_count):
        transition_length = int(trajectory_offsets[trajectory_idx + 1] - trajectory_offsets[trajectory_idx])
        observation_length = int(observation_offsets[trajectory_idx + 1] - observation_offsets[trajectory_idx])
        if observation_length != transition_length + 1:
            raise ValueError(
                f"shard {shard_idx} trajectory {trajectory_idx} must have L+1 observations, "
                f"got {observation_length} for {transition_length} transitions"
            )
    return loaded


def _load_shard(manifest: Mapping[str, Any], shard_idx: int) -> Mapping[str, np.ndarray]:
    path = _shard_path(manifest, shard_idx)
    if not path.is_file():
        raise FileNotFoundError(f"trajectory shard does not exist: {path}")
    if path.suffix != ".npz":
        raise ValueError(f"trajectory shard must be .npz: {path}")
    with np.load(path, allow_pickle=False) as loaded_file:
        loaded = {name: loaded_file[name] for name in loaded_file.files}
    return _validate_shard_arrays(manifest, shard_idx, loaded)


def _trajectory_refs(manifest: Mapping[str, Any], split: str) -> list[Mapping[str, Any]]:
    splits = _split_mapping(manifest)
    if split not in splits:
        raise ValueError(f"unknown dataset split {split!r}; choose one of {sorted(splits)}")
    split_entry = splits[split]
    if not isinstance(split_entry, Mapping):
        raise ValueError(f"manifest split {split!r} must be a mapping")
    refs = split_entry.get("trajectory_refs", split_entry.get("trajectories"))
    if not isinstance(refs, list):
        raise ValueError(f"manifest split {split!r} has no trajectory_refs list")
    return refs


def build_window_index(manifest: Mapping[str, Any], *, split: str) -> WindowIndex:
    """Build overlapping valid windows for one explicit split.

    A candidate is accepted only when all ``K`` transitions and all ``K+1``
    observations are marked valid and no terminal/truncation event occurs in
    the window. This makes endpoint validity independent of the C environment's
    pre-movement eligibility-mask timing.
    """

    validate_manifest(manifest)
    refs = _trajectory_refs(manifest, split)
    chunk_length = int(manifest["chunk_length"])
    windows: list[WindowReference] = []
    shard_cache: dict[int, Mapping[str, np.ndarray]] = {}
    for ref_idx, ref in enumerate(refs):
        if not isinstance(ref, Mapping):
            raise ValueError(f"manifest split {split!r} ref {ref_idx} must be a mapping")
        shard_idx = _as_nonnegative_int(ref.get("shard_idx"), "trajectory ref shard_idx")
        trajectory_idx = _as_nonnegative_int(ref.get("trajectory_idx"), "trajectory ref trajectory_idx")
        if shard_idx not in shard_cache:
            shard_cache[shard_idx] = _load_shard(manifest, shard_idx)
        shard = shard_cache[shard_idx]
        trajectory_offsets = shard["trajectory_offsets"]
        observation_offsets = shard["observation_offsets"]
        if trajectory_idx >= len(trajectory_offsets) - 1:
            raise ValueError(f"trajectory ref {ref_idx} points outside shard {shard_idx}")
        transition_start = int(trajectory_offsets[trajectory_idx])
        transition_end = int(trajectory_offsets[trajectory_idx + 1])
        observation_start = int(observation_offsets[trajectory_idx])
        transition_length = transition_end - transition_start
        if transition_length < chunk_length:
            continue
        valid_transition = shard["valid_transition"][transition_start:transition_end]
        terminated = shard["terminated"][transition_start:transition_end]
        truncated = shard["truncated"][transition_start:transition_end]
        state_valid = shard["state_valid"][observation_start : observation_start + transition_length + 1]
        for local_start in range(transition_length - chunk_length + 1):
            transition_slice = slice(local_start, local_start + chunk_length)
            state_slice = slice(local_start, local_start + chunk_length + 1)
            if not bool(valid_transition[transition_slice].all()):
                continue
            if bool(terminated[transition_slice].any()) or bool(truncated[transition_slice].any()):
                continue
            if not bool(state_valid[state_slice].all()):
                continue
            windows.append(WindowReference(shard_idx, trajectory_idx, local_start))
    return tuple(windows)


class TrajectoryDataset(Dataset):
    """Lazy CPU dataset backed by validated trajectory shards."""

    def __init__(
        self,
        dataset_root: PathLike,
        *,
        split: str,
        manifest: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.dataset_root}")
        if manifest is None:
            manifest_path = self.dataset_root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"manifest.json does not exist below {self.dataset_root}")
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        if not isinstance(manifest, Mapping):
            raise ValueError("manifest must be a mapping")
        manifest_with_root = dict(manifest)
        manifest_with_root["_dataset_root"] = str(self.dataset_root)
        validate_manifest(manifest_with_root)
        self.manifest = manifest_with_root
        self.split = split
        self.window_index = tuple(build_window_index(self.manifest, split=split))
        self._shard_cache: dict[int, Mapping[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.window_index)

    def _get_shard(self, shard_idx: int) -> Mapping[str, np.ndarray]:
        if shard_idx not in self._shard_cache:
            self._shard_cache[shard_idx] = _load_shard(self.manifest, shard_idx)
        return self._shard_cache[shard_idx]

    def __getitem__(self, index: int) -> TrainingSample:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        index_int = int(index)
        if index_int < 0:
            index_int += len(self.window_index)
        if index_int < 0 or index_int >= len(self.window_index):
            raise IndexError(f"window index {index} is out of range for {len(self.window_index)} windows")
        reference = self.window_index[index_int]
        shard = self._get_shard(reference.shard_idx)
        trajectory_start = int(shard["trajectory_offsets"][reference.trajectory_idx])
        observation_start = int(shard["observation_offsets"][reference.trajectory_idx])
        transition_start = trajectory_start + reference.start_step_idx
        sample_observation_start = observation_start + reference.start_step_idx
        chunk_length = int(self.manifest["chunk_length"])
        observations = shard["observations"][sample_observation_start : sample_observation_start + chunk_length + 1]
        controls = shard["executed_controls"][transition_start : transition_start + chunk_length]
        logits = shard["teacher_logits"][transition_start : transition_start + chunk_length]
        if (
            observations.shape[0] != chunk_length + 1
            or controls.shape[0] != chunk_length
            or logits.shape[0] != chunk_length
        ):
            raise ValueError(f"window {reference} is truncated in shard data")
        return TrainingSample(
            observations=torch.from_numpy(np.ascontiguousarray(observations)),
            executed_controls=torch.from_numpy(np.ascontiguousarray(controls)),
            teacher_logits=torch.from_numpy(np.ascontiguousarray(logits)),
        )
