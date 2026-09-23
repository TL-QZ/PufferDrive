"""Bounded-memory Condition B collection and memory-mapped windows.

The legacy collector stores one Python object per trajectory.  This module
keeps a round in time-major ``.npy`` memmaps instead.  A round is deliberately
the unit of storage: its first observation is the live environment state at
the start of the round, and windows never cross that boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .collect import (
    _action_table,
    _as_observation_array,
    _controls_from_teacher,
    _environment_actions,
    _info_has_reset,
    _initial_state,
    _json_safe,
    _nested,
    _observation_layout,
    _physical_action_table,
    _positive_int,
    _reported_masks,
    _step_environment,
    _teacher_checkpoint_identity,
    _teacher_logits,
)
from .dataset import TrainingBatch, TrainingSample


if TYPE_CHECKING:
    from torch.nn import Module


PathLike = Union[str, Path]
_SCHEMA_VERSION = 1
_REQUIRED_SPLITS = ("train", "validation", "test")
_DEFAULT_INFERENCE_BATCH_SIZE = 2048
_NPY_HEADER_RESERVE_BYTES = 4096
_MANIFEST_RESERVE_BYTES = 8192


def _directory_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write the completion marker only after every memmap has been flushed."""

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as output_file:
            json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _array_descriptor(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any], root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "shape": [int(value) for value in shape],
        "dtype": np.dtype(dtype).name,
    }


def _positive_or_default(value: Any, field: str, default: int) -> int:
    if value is None:
        return default
    return _positive_int(value, field)


def _existing_round_transition_count(output_root: Path) -> int:
    """Count completed rounds and reject a stale or crashed round directory."""

    total_transition_count = 0
    if not output_root.exists():
        return total_transition_count
    for round_dir in sorted(output_root.glob("round_*")):
        if not round_dir.is_dir():
            continue
        manifest_path = round_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"streaming collection round is incomplete and cannot be reused: {round_dir}"
            )
        try:
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"streaming collection manifest is incomplete: {manifest_path}") from exc
        if not isinstance(manifest, Mapping) or manifest.get("complete") is not True:
            raise RuntimeError(f"streaming collection manifest is incomplete: {manifest_path}")
        transition_count = manifest.get("collection_transition_count")
        if isinstance(transition_count, bool) or not isinstance(transition_count, (int, np.integer)):
            raise ValueError(f"streaming manifest has invalid transition count: {manifest_path}")
        if int(transition_count) < 0:
            raise ValueError(f"streaming manifest has invalid transition count: {manifest_path}")
        total_transition_count += int(transition_count)
    return total_transition_count


def _observation_dimension_hint(config: Mapping[str, Any], teacher: "Module", env: Any) -> Optional[int]:
    candidates = [
        getattr(teacher, "condition_b_observation_layout", None),
        getattr(teacher, "observation_layout", None),
        _nested(config, "collection").get("observation_layout"),
        getattr(env, "observation_layout", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            value = candidate.get("observation_dim", candidate.get("dim"))
            if value is not None:
                return _positive_int(value, "observation_layout.observation_dim")
    for name in ("_jepa_observations", "observations"):
        value = getattr(env, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] > 0:
            return int(array.shape[1])
    return None


def _slot_count_hint(env: Any) -> Optional[int]:
    value = getattr(env, "num_agents", None)
    if value is not None:
        return _positive_int(value, "env.num_agents")
    for name in ("_jepa_observations", "observations"):
        observations = getattr(env, name, None)
        if observations is None:
            continue
        array = np.asarray(observations)
        if array.ndim == 2 and array.shape[0] > 0:
            return int(array.shape[0])
    return None


def _array_bytes(shape: tuple[int, ...], dtype: np.dtype[Any]) -> int:
    element_count = 1
    for extent in shape:
        if isinstance(extent, bool) or int(extent) < 0:
            raise ValueError("array shape extents must be non-negative integers")
        element_count *= int(extent)
    return element_count * np.dtype(dtype).itemsize


def _estimated_round_bytes(
    *,
    transition_steps: int,
    slots: int,
    observation_dim: int,
    classes: int,
    chunk_length: int,
) -> int:
    """Conservative uncompressed byte count used before reset or stepping."""

    observation_shape = (transition_steps + 1, slots, observation_dim)
    transition_shape = (transition_steps, slots)
    endpoint_shape = (transition_steps + 1, slots)
    total = 0
    array_specs = (
        (observation_shape, np.dtype(np.float32)),
        ((transition_steps, slots, 2), np.dtype(np.float32)),
        ((transition_steps, slots, classes), np.dtype(np.float32)),
        (transition_shape, np.dtype(np.bool_)),
        (transition_shape, np.dtype(np.bool_)),
        (transition_shape, np.dtype(np.bool_)),
        (transition_shape, np.dtype(np.bool_)),
        (endpoint_shape, np.dtype(np.bool_)),
        (endpoint_shape, np.dtype(np.int64)),
        ((max(0, transition_steps - chunk_length + 1) * slots,), np.dtype(np.int64)),
    )
    for shape, dtype in array_specs:
        total += _array_bytes(tuple(int(value) for value in shape), dtype) + _NPY_HEADER_RESERVE_BYTES
    return total + _MANIFEST_RESERVE_BYTES


def _teacher_logits_chunked(
    teacher: "Module",
    observations: np.ndarray,
    classes: int,
    inference_batch_size: int,
) -> torch.Tensor:
    """Run the legacy logits helper in bounded inference chunks."""

    logits_chunks: list[torch.Tensor] = []
    for start_idx in range(0, observations.shape[0], inference_batch_size):
        end_idx = min(start_idx + inference_batch_size, observations.shape[0])
        logits_chunks.append(_teacher_logits(teacher, observations[start_idx:end_idx], classes))
    if not logits_chunks:
        raise ValueError("teacher inference received no environment slots")
    logits = torch.cat(logits_chunks, dim=0)
    if logits.shape != (observations.shape[0], classes):
        raise ValueError(
            f"chunked teacher logits must have shape [{observations.shape[0]},{classes}], got {tuple(logits.shape)}"
        )
    return logits


def _window_validity(
    transition_valid: np.ndarray,
    endpoint_valid: np.ndarray,
    generations: np.ndarray,
    chunk_length: int,
    *,
    output_indices: Optional[np.ndarray] = None,
) -> int:
    """Count or write valid flattened ``time * slots + slot`` starts.

    The rolling sums are cumulative invalid counts for the transition and
    endpoint windows.  Each row is processed while only one vector-sized mask
    is resident, so the routine does not build a Python tuple per window.
    """

    transition_steps, slots = transition_valid.shape
    candidate_steps = max(0, transition_steps - chunk_length + 1)
    if candidate_steps == 0:
        return 0
    if endpoint_valid.shape != (transition_steps + 1, slots):
        raise ValueError("endpoint validity shape does not match transition validity")
    if generations.shape != (transition_steps + 1, slots):
        raise ValueError("generation shape does not match transition validity")

    transition_invalid_count = np.zeros(slots, dtype=np.int64)
    endpoint_invalid_count = np.zeros(slots, dtype=np.int64)
    for transition_idx in range(chunk_length):
        transition_invalid_count += (
            ~np.asarray(transition_valid[transition_idx], dtype=bool)
            | (generations[transition_idx + 1] != generations[transition_idx])
        )
    for endpoint_idx in range(chunk_length + 1):
        endpoint_invalid_count += ~np.asarray(endpoint_valid[endpoint_idx], dtype=bool)
    write_offset = 0
    valid_count = 0
    for start_idx in range(candidate_steps):
        if start_idx:
            added_transition_idx = start_idx + chunk_length - 1
            removed_transition_idx = start_idx - 1
            transition_invalid_count += (
                ~np.asarray(transition_valid[added_transition_idx], dtype=bool)
                | (generations[added_transition_idx + 1] != generations[added_transition_idx])
            )
            transition_invalid_count -= (
                ~np.asarray(transition_valid[removed_transition_idx], dtype=bool)
                | (generations[removed_transition_idx + 1] != generations[removed_transition_idx])
            )
            endpoint_invalid_count += ~np.asarray(endpoint_valid[start_idx + chunk_length], dtype=bool)
            endpoint_invalid_count -= ~np.asarray(endpoint_valid[start_idx - 1], dtype=bool)
        valid_slots = np.flatnonzero(
            (transition_invalid_count == 0) & (endpoint_invalid_count == 0)
        ).astype(np.int64, copy=False)
        valid_slots_count = int(valid_slots.size)
        if output_indices is not None and valid_slots_count:
            output_indices[write_offset : write_offset + valid_slots_count] = (
                np.asarray(start_idx, dtype=np.int64) * slots + valid_slots
            )
            write_offset += valid_slots_count
        valid_count += valid_slots_count
    if output_indices is not None and write_offset != valid_count:
        raise RuntimeError("streaming window index write count disagrees with validity count")
    return valid_count


def _load_streaming_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"streaming manifest does not exist: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"streaming manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("streaming manifest must be a mapping")
    if manifest.get("complete") is not True:
        raise ValueError(f"streaming manifest is incomplete: {manifest_path}")
    if manifest.get("dataset_format") != "condition_b_streaming_npy_v1":
        raise ValueError("unsupported streaming dataset format")
    return manifest


def _manifest_array_path(manifest_path: Path, entry: Any, field: str) -> tuple[Path, Optional[tuple[int, ...]], Optional[np.dtype[Any]]]:
    if isinstance(entry, str):
        relative_path = entry
        shape = None
        dtype = None
    elif isinstance(entry, Mapping):
        relative_path = entry.get("path")
        raw_shape = entry.get("shape")
        shape = tuple(int(value) for value in raw_shape) if raw_shape is not None else None
        raw_dtype = entry.get("dtype")
        dtype = np.dtype(raw_dtype) if raw_dtype is not None else None
    else:
        raise ValueError(f"streaming manifest arrays.{field} must be a path or descriptor")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"streaming manifest arrays.{field}.path must be a non-empty string")
    path = Path(relative_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ValueError(f"streaming manifest arrays.{field} points outside its dataset root") from exc
    return resolved_path, shape, dtype


class StreamingTrajectoryDataset(Dataset):
    """Lazy dataset backed by one round's time-major NumPy memmaps."""

    def __init__(self, manifest_path: PathLike, *, split: Optional[str] = None) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = _load_streaming_manifest(self.manifest_path)
        if split is not None and self.manifest.get("split") != split:
            raise ValueError(
                f"streaming manifest belongs to split {self.manifest.get('split')!r}, expected {split!r}"
            )
        if split is None:
            split = self.manifest.get("split")
        if not isinstance(split, str) or not split:
            raise ValueError("streaming manifest split must be a non-empty string")
        self.split = split

        arrays = self.manifest.get("arrays")
        if not isinstance(arrays, Mapping):
            raise ValueError("streaming manifest arrays must be a mapping")
        self._array_descriptors: dict[str, tuple[Path, Optional[tuple[int, ...]], Optional[np.dtype[Any]]]] = {}
        for name in (
            "observations",
            "controls",
            "logits",
            "transition_valid",
            "eligibility_mask",
            "terminated",
            "truncated",
            "endpoint_valid",
            "generation",
        ):
            if name not in arrays:
                raise ValueError(f"streaming manifest is missing arrays.{name}")
            self._array_descriptors[name] = _manifest_array_path(self.manifest_path, arrays[name], name)

        window_entry = self.manifest.get("window_indices")
        if window_entry is None:
            raise ValueError("streaming manifest is missing window_indices")
        window_path, window_shape, window_dtype = _manifest_array_path(
            self.manifest_path, window_entry, "window_indices"
        )
        self._array_descriptors["window_indices"] = (window_path, window_shape, window_dtype)

        chunk_length = self.manifest.get("chunk_length")
        slots = self.manifest.get("slots")
        observation_dim = self.manifest.get("observation_dim")
        classes = self.manifest.get("num_action_classes")
        for field, value in (
            ("chunk_length", chunk_length),
            ("slots", slots),
            ("observation_dim", observation_dim),
            ("num_action_classes", classes),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"streaming manifest {field} must be a positive integer")
        self.chunk_length = int(chunk_length)
        self.slots = int(slots)
        self.observation_dim = int(observation_dim)
        self.num_action_classes = int(classes)
        transition_steps = self.manifest.get("simulator_steps")
        if isinstance(transition_steps, bool) or not isinstance(transition_steps, (int, np.integer)):
            raise ValueError("streaming manifest simulator_steps must be a non-negative integer")
        self.transition_steps = int(transition_steps)

        self.observations = self._open_array("observations", (self.transition_steps + 1, self.slots, self.observation_dim), np.float32)
        self.controls = self._open_array("controls", (self.transition_steps, self.slots, 2), np.float32)
        self.teacher_logits = self._open_array("logits", (self.transition_steps, self.slots, self.num_action_classes), np.float32)
        self.transition_valid = self._open_array("transition_valid", (self.transition_steps, self.slots), np.bool_)
        self.eligibility_mask = self._open_array("eligibility_mask", (self.transition_steps, self.slots), np.bool_)
        self.terminated = self._open_array("terminated", (self.transition_steps, self.slots), np.bool_)
        self.truncated = self._open_array("truncated", (self.transition_steps, self.slots), np.bool_)
        self.endpoint_valid = self._open_array("endpoint_valid", (self.transition_steps + 1, self.slots), np.bool_)
        self.generations = self._open_array("generation", (self.transition_steps + 1, self.slots), np.int64)

        window_path, expected_shape, expected_dtype = self._array_descriptors["window_indices"]
        if not window_path.is_file():
            raise FileNotFoundError(f"streaming window index does not exist: {window_path}")
        self.window_indices = np.load(window_path, mmap_mode="r", allow_pickle=False)
        if self.window_indices.ndim != 1 or self.window_indices.dtype != np.int64:
            raise ValueError("streaming window_indices must be one-dimensional int64")
        if expected_shape is not None and tuple(self.window_indices.shape) != expected_shape:
            raise ValueError("streaming window_indices shape disagrees with manifest")
        if expected_dtype is not None and self.window_indices.dtype != expected_dtype:
            raise ValueError("streaming window_indices dtype disagrees with manifest")
        declared_window_count = self.manifest.get("valid_window_count")
        if declared_window_count is not None and int(declared_window_count) != len(self.window_indices):
            raise ValueError("streaming valid_window_count disagrees with window index array")

    def _open_array(self, name: str, expected_shape: tuple[int, ...], expected_dtype: np.dtype[Any]) -> np.ndarray:
        path, declared_shape, declared_dtype = self._array_descriptors[name]
        if not path.is_file():
            raise FileNotFoundError(f"streaming array {name} does not exist: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != expected_shape or array.dtype != expected_dtype:
            raise ValueError(
                f"streaming array {name} has shape/dtype {array.shape}/{array.dtype}, "
                f"expected {expected_shape}/{expected_dtype}"
            )
        if declared_shape is not None and tuple(array.shape) != declared_shape:
            raise ValueError(f"streaming array {name} shape disagrees with manifest")
        if declared_dtype is not None and array.dtype != declared_dtype:
            raise ValueError(f"streaming array {name} dtype disagrees with manifest")
        return array

    def __len__(self) -> int:
        return int(self.window_indices.shape[0])

    def _normalize_index(self, index: Any) -> int:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        index_int = int(index)
        if index_int < 0:
            index_int += len(self)
        if index_int < 0 or index_int >= len(self):
            raise IndexError(f"window index {index} is out of range for {len(self)} windows")
        return index_int

    def __getitem__(self, index: int) -> TrainingSample:
        index_int = self._normalize_index(index)
        flattened_start = int(self.window_indices[index_int])
        transition_start = flattened_start // self.slots
        slot_idx = flattened_start % self.slots
        observations = self.observations[
            transition_start : transition_start + self.chunk_length + 1, slot_idx, :
        ]
        controls = self.controls[transition_start : transition_start + self.chunk_length, slot_idx, :]
        logits = self.teacher_logits[transition_start : transition_start + self.chunk_length, slot_idx, :]
        if (
            observations.shape != (self.chunk_length + 1, self.observation_dim)
            or controls.shape != (self.chunk_length, 2)
            or logits.shape != (self.chunk_length, self.num_action_classes)
        ):
            raise ValueError(f"streaming window {index} is truncated in memmap data")
        return TrainingSample(
            observations=torch.from_numpy(np.ascontiguousarray(observations)),
            executed_controls=torch.from_numpy(np.ascontiguousarray(controls)),
            teacher_logits=torch.from_numpy(np.ascontiguousarray(logits)),
        )

    def _normalize_batch_indices(self, indices: Any) -> np.ndarray:
        if isinstance(indices, torch.Tensor):
            if indices.dtype == torch.bool or indices.dtype.is_floating_point:
                raise TypeError("batch indices must be integers")
            indices = indices.detach().cpu().numpy()
        raw_indices = np.asarray(indices)
        if raw_indices.ndim == 0:
            raw_indices = raw_indices.reshape(1)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise TypeError("batch indices must be a one-dimensional integer array")
        normalized = raw_indices.astype(np.int64, copy=True)
        normalized[normalized < 0] += len(self)
        if np.any(normalized < 0) or np.any(normalized >= len(self)):
            raise IndexError(f"batch index is out of range for {len(self)} windows")
        return normalized

    def get_batch(self, indices: Any) -> TrainingBatch:
        """Fetch a caller-sized microbatch with vectorized memmap indexing."""

        normalized = self._normalize_batch_indices(indices)
        flattened_starts = np.asarray(self.window_indices[normalized], dtype=np.int64)
        transition_starts = flattened_starts // self.slots
        slot_indices = flattened_starts % self.slots
        time_offsets = np.arange(self.chunk_length + 1, dtype=np.int64)
        transition_offsets = np.arange(self.chunk_length, dtype=np.int64)
        observation_values = np.ascontiguousarray(
            self.observations[transition_starts[:, None] + time_offsets[None, :], slot_indices[:, None], :]
        )
        control_values = np.ascontiguousarray(
            self.controls[transition_starts[:, None] + transition_offsets[None, :], slot_indices[:, None], :]
        )
        logit_values = np.ascontiguousarray(
            self.teacher_logits[transition_starts[:, None] + transition_offsets[None, :], slot_indices[:, None], :]
        )
        return TrainingBatch(
            observations=torch.from_numpy(observation_values),
            executed_controls=torch.from_numpy(control_values),
            teacher_logits=torch.from_numpy(logit_values),
        )


def collect_streaming_dataset(
    config: Mapping[str, Any],
    output_dir: PathLike,
    *,
    teacher: "Module",
    env: Any,
    collection_round_idx: int,
) -> Mapping[str, Any]:
    """Collect one live environment round into bounded-memory memmaps."""

    if isinstance(collection_round_idx, bool) or int(collection_round_idx) < 0:
        raise ValueError("collection_round_idx must be a non-negative integer")
    collection_round_idx = int(collection_round_idx)
    collection = _nested(config, "collection")
    model = _nested(config, "model")
    chunk_length = _positive_int(
        model.get("chunk_length", collection.get("chunk_length", 4)), "model.chunk_length"
    )
    classes = _positive_int(model.get("num_action_classes", 12), "model.num_action_classes")
    requested_transitions = _positive_int(
        collection.get("transitions_per_round"), "collection.transitions_per_round"
    )
    split = collection.get("split", "train")
    if not isinstance(split, str) or not split:
        raise ValueError("collection.split must be a non-empty string")
    if split not in _REQUIRED_SPLITS:
        raise ValueError(f"collection.split must be one of {_REQUIRED_SPLITS}, got {split!r}")
    action_selection = collection.get("action_selection", "sample")
    if action_selection not in ("sample", "mean", "mode"):
        raise ValueError("collection.action_selection must be 'sample', 'mean', or 'mode'")
    inference_batch_size = _positive_or_default(
        collection.get("inference_batch_size"),
        "collection.inference_batch_size",
        _DEFAULT_INFERENCE_BATCH_SIZE,
    )

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    round_dir = output_root / f"round_{collection_round_idx:04d}"
    if round_dir.exists():
        raise FileExistsError(
            f"streaming collection round output exists (possibly incomplete; refusing reuse): {round_dir}"
        )

    slots_hint = _slot_count_hint(env)
    observation_dim_hint = _observation_dimension_hint(config, teacher, env)
    if slots_hint is None:
        raise ValueError("streaming collector needs env.num_agents or existing observations before preflight")
    max_transitions = collection.get("max_transitions")
    existing_transition_count = _existing_round_transition_count(output_root)
    if max_transitions is not None:
        max_transitions = _positive_int(max_transitions, "collection.max_transitions")
        remaining_transitions = max_transitions - existing_transition_count
        requested_transitions = min(requested_transitions, max(0, remaining_transitions))
        requested_transitions = (requested_transitions // slots_hint) * slots_hint
        if requested_transitions <= 0:
            raise RuntimeError("collection.max_transitions has no complete vector step remaining")
    if requested_transitions % slots_hint:
        raise ValueError(
            "collection.transitions_per_round must be a multiple of env.num_agents "
            "so every vector slot has the same number of outcomes"
        )
    simulator_steps = requested_transitions // slots_hint
    if simulator_steps < 1:
        raise ValueError("collection.transitions_per_round must include at least one vector step")

    max_disk_bytes = collection.get("max_disk_bytes")
    if max_disk_bytes is not None:
        max_disk_bytes = _positive_int(max_disk_bytes, "collection.max_disk_bytes")
    if observation_dim_hint is not None:
        estimated_bytes = _estimated_round_bytes(
            transition_steps=simulator_steps,
            slots=slots_hint,
            observation_dim=observation_dim_hint,
            classes=classes,
            chunk_length=chunk_length,
        )
        existing_bytes = _directory_size(output_root)
        if max_disk_bytes is not None and existing_bytes + estimated_bytes > max_disk_bytes:
            raise RuntimeError(
                "collection.max_disk_bytes is too small for the requested streaming round "
                f"(minimum projected {existing_bytes + estimated_bytes} > {max_disk_bytes})"
            )
        free_bytes = shutil.disk_usage(output_root).free
        if free_bytes < estimated_bytes:
            raise RuntimeError(
                "insufficient free disk space for streaming round "
                f"(required at least {estimated_bytes}, free {free_bytes})"
            )

    observations, generations, active, _ = _initial_state(env, config, split)
    slots, observation_dim = observations.shape
    if slots != slots_hint:
        raise ValueError(f"env.num_agents={slots_hint} disagrees with initial observations slots={slots}")
    env_num_agents = getattr(env, "num_agents", slots)
    if int(env_num_agents) != slots:
        raise ValueError(f"env.num_agents={env_num_agents} disagrees with observations slots={slots}")
    if observation_dim_hint is not None and observation_dim != observation_dim_hint:
        raise ValueError(
            f"initial observation width {observation_dim} disagrees with layout hint {observation_dim_hint}"
        )
    if max_disk_bytes is None or observation_dim_hint is None:
        estimated_bytes = _estimated_round_bytes(
            transition_steps=simulator_steps,
            slots=slots,
            observation_dim=observation_dim,
            classes=classes,
            chunk_length=chunk_length,
        )
        existing_bytes = _directory_size(output_root)
        if max_disk_bytes is not None and existing_bytes + estimated_bytes > max_disk_bytes:
            raise RuntimeError(
                "collection.max_disk_bytes is too small for the requested streaming round "
                f"(minimum projected {existing_bytes + estimated_bytes} > {max_disk_bytes})"
            )
        free_bytes = shutil.disk_usage(output_root).free
        if free_bytes < estimated_bytes:
            raise RuntimeError(
                "insufficient free disk space for streaming round "
                f"(required at least {estimated_bytes}, free {free_bytes})"
            )

    layout = _observation_layout(config, teacher, env, observation_dim)
    round_dir.mkdir(parents=True, exist_ok=False)
    array_shapes: dict[str, tuple[int, ...]] = {
        "observations": (simulator_steps + 1, slots, observation_dim),
        "controls": (simulator_steps, slots, 2),
        "logits": (simulator_steps, slots, classes),
        "transition_valid": (simulator_steps, slots),
        "eligibility_mask": (simulator_steps, slots),
        "terminated": (simulator_steps, slots),
        "truncated": (simulator_steps, slots),
        "endpoint_valid": (simulator_steps + 1, slots),
        "generation": (simulator_steps + 1, slots),
    }
    array_dtypes: dict[str, np.dtype[Any]] = {
        "observations": np.dtype(np.float32),
        "controls": np.dtype(np.float32),
        "logits": np.dtype(np.float32),
        "transition_valid": np.dtype(np.bool_),
        "eligibility_mask": np.dtype(np.bool_),
        "terminated": np.dtype(np.bool_),
        "truncated": np.dtype(np.bool_),
        "endpoint_valid": np.dtype(np.bool_),
        "generation": np.dtype(np.int64),
    }
    memmaps: dict[str, np.memmap] = {}
    array_descriptors: dict[str, dict[str, Any]] = {}
    for name, shape in array_shapes.items():
        path = round_dir / f"{name}.npy"
        memmaps[name] = np.lib.format.open_memmap(path, mode="w+", dtype=array_dtypes[name], shape=shape)
        array_descriptors[name] = _array_descriptor(path, shape, array_dtypes[name], round_dir)

    memmaps["observations"][0] = observations
    memmaps["endpoint_valid"][0] = active & np.isfinite(observations).all(axis=1)
    memmaps["generation"][0] = generations
    rejected_reasons: dict[str, int] = {}
    valid_transition_count = 0
    stored_transition_count = 0
    candidate_window_count = 0
    segment_transition_lengths = np.zeros(slots, dtype=np.int64)
    mask_available = False
    started_at = time.monotonic()

    for timestep in range(simulator_steps):
        logits = _teacher_logits_chunked(teacher, observations, classes, inference_batch_size)
        action_ids, controls = _controls_from_teacher(teacher, logits, action_selection, config)
        environment_actions = _environment_actions(env, action_ids, controls)
        (
            next_observations,
            _,
            terminated,
            truncated,
            info,
            returned_masks,
        ) = _step_environment(env, environment_actions)
        if next_observations.shape != observations.shape:
            raise ValueError("environment observation shape changed within a streaming collection round")
        if returned_masks is None:
            reported_masks, has_masks = _reported_masks(env, slots)
        else:
            reported_masks, has_masks = returned_masks, True
        mask_available = mask_available or has_masks
        reset_event = _info_has_reset(info)
        active_before = active.copy()
        segment_transition_lengths[active_before] += 1
        current_finite = np.isfinite(observations).all(axis=1)
        endpoint_finite = np.isfinite(next_observations).all(axis=1)
        boundary = terminated | truncated
        pre_action_eligible = np.asarray(reported_masks, dtype=bool)
        valid = active & pre_action_eligible & ~boundary & current_finite & endpoint_finite
        reset_start = truncated | (terminated & reset_event)

        memmaps["controls"][timestep] = controls
        memmaps["logits"][timestep] = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        memmaps["transition_valid"][timestep] = valid
        memmaps["eligibility_mask"][timestep] = pre_action_eligible
        memmaps["terminated"][timestep] = terminated
        memmaps["truncated"][timestep] = truncated
        memmaps["observations"][timestep + 1] = next_observations

        endpoint_valid = valid | reset_start
        next_generations = generations.copy()
        next_generations[reset_start] += 1
        memmaps["endpoint_valid"][timestep + 1] = endpoint_valid
        memmaps["generation"][timestep + 1] = next_generations

        valid_transition_count += int(valid.sum())
        stored_transition_count += int(active.sum())
        reason_masks = (
            ("inactive_slot", ~active),
            ("ineligible_mask", ~pre_action_eligible),
            ("terminated", terminated),
            ("truncated", truncated),
            ("nonfinite_observation", ~(current_finite & endpoint_finite)),
        )
        reason_seen = False
        for reason, reason_mask in reason_masks:
            reason_count = int(reason_mask.sum())
            if reason_count:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + reason_count
                reason_seen = True
        invalid_count = slots - int(valid.sum())
        if invalid_count and not reason_seen:
            rejected_reasons["invalid_transition"] = rejected_reasons.get("invalid_transition", 0) + invalid_count

        segment_closed = (~pre_action_eligible) | boundary | (~active_before)
        closed_lengths = segment_transition_lengths[segment_closed]
        if closed_lengths.size:
            candidate_window_count += int(
                np.maximum(0, closed_lengths - chunk_length + 1).sum()
            )
        segment_transition_lengths[segment_closed] = 0
        active = valid | reset_start
        generations = next_generations
        observations = np.ascontiguousarray(next_observations.copy())

    candidate_window_count += int(
        np.maximum(0, segment_transition_lengths - chunk_length + 1).sum()
    )
    for memmap in memmaps.values():
        memmap.flush()
    setattr(env, "_jepa_observations", observations.copy())
    setattr(
        env,
        "_jepa_collector_state",
        {
            "reset_generation": generations.astype(np.int64).tolist(),
            "slot_active": active.astype(bool).tolist(),
        },
    )

    valid_window_count = _window_validity(
        memmaps["transition_valid"],
        memmaps["endpoint_valid"],
        memmaps["generation"],
        chunk_length,
    )
    window_path = round_dir / "window_indices.npy"
    window_indices = np.lib.format.open_memmap(
        window_path,
        mode="w+",
        dtype=np.int64,
        shape=(valid_window_count,),
    )
    _window_validity(
        memmaps["transition_valid"],
        memmaps["endpoint_valid"],
        memmaps["generation"],
        chunk_length,
        output_indices=window_indices,
    )
    window_indices.flush()
    window_descriptor = _array_descriptor(window_path, (valid_window_count,), np.dtype(np.int64), round_dir)

    checkpoint_path, checkpoint_sha256 = _teacher_checkpoint_identity(teacher, config)
    physical_action_table = _physical_action_table(teacher, config, classes)
    action_layout: dict[str, Any] = {
        "normalized_controls": _action_table(teacher, config, classes).tolist(),
        "control_order": ["longitudinal", "lateral"],
        "normalized_range": [-1.0, 1.0],
    }
    if physical_action_table is not None:
        action_layout["physical_controls"] = physical_action_table.tolist()
    split_seed = (
        collection.get("split_seeds", {}).get(split)
        if isinstance(collection.get("split_seeds"), Mapping)
        else None
    )
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset_format": "condition_b_streaming_npy_v1",
        "complete": True,
        "dataset_root": str(round_dir),
        "collection_round_idx": collection_round_idx,
        "split": split,
        "collection_mode": "teacher_driven_streaming",
        "action_selection": action_selection,
        "chunk_length": chunk_length,
        "num_action_classes": classes,
        "slots": slots,
        "observation_dim": observation_dim,
        "simulator_steps": simulator_steps,
        "observation_layout": layout,
        "action_layout": action_layout,
        "teacher_checkpoint": checkpoint_path,
        "teacher_checkpoint_sha256": checkpoint_sha256,
        "split_seed": split_seed,
        "effective_config": _json_safe(config),
        "code_revision": os.environ.get("GIT_COMMIT", config.get("code_revision")),
        "arrays": array_descriptors,
        "window_indices": window_descriptor,
        "collection_transition_count": requested_transitions,
        "transition_count": requested_transitions,
        "valid_window_count": valid_window_count,
        "candidate_window_count": candidate_window_count,
        "rejected_window_count": candidate_window_count - valid_window_count,
        "eligibility_mask_timing": "action_pre_movement",
        "endpoint_validity": "finite_next_observation_and_lifecycle_events",
        "window_index_flattening": "time_index * slots + slot_index",
        "stats": {
            "requested_transition_count": requested_transitions,
            "collected_transition_count": requested_transitions,
            "simulator_steps": simulator_steps,
            "stored_transition_count": stored_transition_count,
            "valid_transition_count": valid_transition_count,
            "rejected_transition_count": requested_transitions - valid_transition_count,
            "valid_window_count": valid_window_count,
            "candidate_window_count": candidate_window_count,
            "rejected_window_count": candidate_window_count - valid_window_count,
            "rejected_reasons": rejected_reasons,
            "eligibility_mask_available": mask_available,
            "collection_seconds": time.monotonic() - started_at,
            "inference_batch_size": inference_batch_size,
        },
        "cost": {
            "simulator_steps": simulator_steps,
            "agent_transitions": requested_transitions,
            "stored_transitions": stored_transition_count,
            "collection_seconds": time.monotonic() - started_at,
            "storage_bytes_before_manifest": _directory_size(round_dir),
        },
    }
    if max_disk_bytes is not None:
        current_size = _directory_size(output_root)
        manifest_bytes = len(json.dumps(manifest, sort_keys=True).encode("utf-8")) + _MANIFEST_RESERVE_BYTES
        if current_size + manifest_bytes > max_disk_bytes:
            raise RuntimeError(
                "collection.max_disk_bytes would be exceeded before writing streaming manifest "
                f"(projected {current_size + manifest_bytes} > {max_disk_bytes})"
            )
    manifest_path = round_dir / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    result = dict(manifest)
    result["manifest_path"] = str(manifest_path)
    return result


__all__ = ["StreamingTrajectoryDataset", "collect_streaming_dataset"]
