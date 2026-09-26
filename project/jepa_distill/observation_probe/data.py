"""Header-only preflight for saved Condition B streaming collections."""
from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..runtime import resolve_path

if TYPE_CHECKING:
    from .contracts import ProbeBatch


_DATASET_FORMAT = "condition_b_streaming_npy_v1"
_MAX_NPY_HEADER_SIZE = 1_000_000
_SHUFFLE_BLOCK_SIZE = 65_536
_ARRAY_LAYOUTS = {
    "observations": ("float32", lambda steps, slots, dim, classes: (steps + 1, slots, dim)),
    "controls": ("float32", lambda steps, slots, dim, classes: (steps, slots, 2)),
    "logits": ("float32", lambda steps, slots, dim, classes: (steps, slots, classes)),
    "transition_valid": ("bool", lambda steps, slots, dim, classes: (steps, slots)),
    "eligibility_mask": ("bool", lambda steps, slots, dim, classes: (steps, slots)),
    "terminated": ("bool", lambda steps, slots, dim, classes: (steps, slots)),
    "truncated": ("bool", lambda steps, slots, dim, classes: (steps, slots)),
    "endpoint_valid": ("bool", lambda steps, slots, dim, classes: (steps + 1, slots)),
    "generation": ("int64", lambda steps, slots, dim, classes: (steps + 1, slots)),
}
_TEACHER_ENV_KEYS = {"action_type", "dt", "dynamics_model", "reward_conditioning"}


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _resolve_inside(path_text: Any, directory: Path, field: str) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise ValueError(f"{field}.path must be a non-empty string")
    path = Path(path_text).expanduser()
    if path.is_absolute():
        candidate = path
    else:
        candidate = directory / path
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(directory.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{field} points outside its collection directory: {path_text}") from exc
    return resolved


def _read_array_header(path: Path, field: str) -> tuple[tuple[int, ...], np.dtype[Any], int]:
    """Read only the NPY header and check the declared file byte length."""

    if not path.is_file():
        raise FileNotFoundError(f"collection array {field} does not exist: {path}")
    try:
        with path.open("rb") as array_file:
            version = np.lib.format.read_magic(array_file)
            if version == (1, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    array_file, max_header_size=_MAX_NPY_HEADER_SIZE
                )
            elif version == (2, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    array_file, max_header_size=_MAX_NPY_HEADER_SIZE
                )
            else:
                raise ValueError(f"unsupported NPY format version {version}")
            data_offset = array_file.tell()
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"collection array {field} has an invalid NPY header: {path}") from exc
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError(f"collection array {field} must not contain Python objects")
    element_count = 1
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError(f"collection array {field} has an invalid shape")
        element_count *= extent
    expected_bytes = data_offset + element_count * dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"collection array {field} byte size disagrees with its NPY header: {path}"
        )
    return tuple(shape), dtype, data_offset


def _inspect_descriptor(
    entry: Any,
    *,
    manifest_directory: Path,
    field: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> Path:
    if not isinstance(entry, Mapping):
        raise ValueError(f"manifest {field} descriptor must be a mapping")
    path = _resolve_inside(entry.get("path"), manifest_directory, field)
    descriptor_shape = entry.get("shape")
    if not isinstance(descriptor_shape, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in descriptor_shape
    ):
        raise ValueError(f"manifest {field}.shape must be a list of non-negative integers")
    try:
        descriptor_dtype = np.dtype(entry.get("dtype"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest {field}.dtype is invalid") from exc

    shape, dtype, _ = _read_array_header(path, field)
    required_dtype = np.dtype(expected_dtype)
    if shape != expected_shape or tuple(descriptor_shape) != expected_shape:
        raise ValueError(f"manifest/NPY shape mismatch for {field}: expected {expected_shape}")
    if dtype != required_dtype or descriptor_dtype != required_dtype:
        raise ValueError(f"manifest/NPY dtype mismatch for {field}: expected {required_dtype}")
    return path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _teacher_observation_recipe(teacher_config: Any) -> dict[str, Any] | None:
    if not isinstance(teacher_config, Mapping):
        return None
    environment = teacher_config.get("env")
    if not isinstance(environment, Mapping):
        return None
    recipe = {
        key: value
        for key, value in environment.items()
        if key in _TEACHER_ENV_KEYS or key.startswith("obs_norm_")
    }
    return recipe or None


def _validate_action_layout(action_layout: Any, classes: int, manifest_path: Path) -> None:
    if not isinstance(action_layout, Mapping):
        raise ValueError(f"manifest action_layout must be a mapping: {manifest_path}")
    if action_layout.get("control_order") != ["longitudinal", "lateral"]:
        raise ValueError(f"unsupported action control order: {manifest_path}")
    controls = action_layout.get("normalized_controls")
    if not isinstance(controls, list) or len(controls) != classes:
        raise ValueError(f"action_layout normalized_controls disagrees with action classes: {manifest_path}")
    for action_idx, control in enumerate(controls):
        if not isinstance(control, list) or len(control) != 2:
            raise ValueError(f"action_layout normalized control {action_idx} must have two values")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in control
        ):
            raise ValueError(f"action_layout normalized control {action_idx} has invalid values")


def _inspect_manifest(
    manifest_path: Path,
    *,
    expected_round_idx: int | None = None,
    expected_split: str = "train",
) -> dict[str, Any]:
    if expected_split not in {"train", "validation", "test"}:
        raise ValueError("expected_split must be 'train', 'validation', or 'test'")
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"streaming manifest must be a mapping: {manifest_path}")
    if manifest.get("complete") is not True:
        raise ValueError(f"streaming manifest is incomplete: {manifest_path}")
    if manifest.get("dataset_format") != _DATASET_FORMAT:
        raise ValueError(f"unsupported streaming manifest format: {manifest_path}")
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or schema_version not in (1, "1"):
        raise ValueError(f"unsupported streaming manifest schema_version {schema_version!r}")
    if manifest.get("split") != expected_split:
        raise ValueError(
            f"streaming manifest belongs to split {manifest.get('split')!r}, "
            f"expected {expected_split!r}: {manifest_path}"
        )
    round_idx = _nonnegative_int(manifest.get("collection_round_idx"), "collection_round_idx")
    if expected_round_idx is not None and round_idx != expected_round_idx:
        raise ValueError(f"manifest round {round_idx} does not match selected round {expected_round_idx}")

    steps = _positive_int(manifest.get("simulator_steps"), "simulator_steps")
    slots = _positive_int(manifest.get("slots"), "slots")
    observation_dim = _positive_int(manifest.get("observation_dim"), "observation_dim")
    chunk_length = _positive_int(manifest.get("chunk_length"), "chunk_length")
    action_classes = _positive_int(manifest.get("num_action_classes"), "num_action_classes")
    valid_window_count = _nonnegative_int(manifest.get("valid_window_count"), "valid_window_count")
    if valid_window_count > steps * slots:
        raise ValueError(f"manifest valid_window_count exceeds candidate starts: {manifest_path}")

    layout = manifest.get("observation_layout")
    if not isinstance(layout, Mapping):
        raise ValueError(f"manifest observation_layout must be a mapping: {manifest_path}")
    layout_dim = layout.get("observation_dim", layout.get("dim"))
    if isinstance(layout_dim, bool) or not isinstance(layout_dim, int) or layout_dim != observation_dim:
        raise ValueError(f"manifest observation layout width disagrees with observation_dim: {manifest_path}")
    action_layout = manifest.get("action_layout")
    _validate_action_layout(action_layout, action_classes, manifest_path)
    effective_config = manifest.get("effective_config")
    teacher_config = (
        effective_config.get("teacher_config")
        if isinstance(effective_config, Mapping)
        else None
    )
    teacher_observation_recipe = _teacher_observation_recipe(teacher_config)

    array_descriptors = manifest.get("arrays")
    if not isinstance(array_descriptors, Mapping):
        raise ValueError(f"manifest arrays must be a mapping: {manifest_path}")
    expected_dimensions = {
        name: shape_fn(steps, slots, observation_dim, action_classes)
        for name, (_dtype, shape_fn) in _ARRAY_LAYOUTS.items()
    }
    array_paths: dict[str, str] = {}
    for name, (expected_dtype, _shape_fn) in _ARRAY_LAYOUTS.items():
        if name not in array_descriptors:
            raise ValueError(f"manifest is missing arrays.{name}: {manifest_path}")
        array_path = _inspect_descriptor(
            array_descriptors[name],
            manifest_directory=manifest_path.parent,
            field=f"arrays.{name}",
            expected_shape=expected_dimensions[name],
            expected_dtype=expected_dtype,
        )
        array_paths[name] = str(array_path)

    window_path = _inspect_descriptor(
        manifest.get("window_indices"),
        manifest_directory=manifest_path.parent,
        field="window_indices",
        expected_shape=(valid_window_count,),
        expected_dtype="int64",
    )

    return {
        "manifest_path": str(manifest_path),
        "split": expected_split,
        "collection_round_idx": round_idx,
        "split_seed": manifest.get("split_seed"),
        "simulator_steps": steps,
        "collection_transition_count": manifest.get(
            "collection_transition_count", manifest.get("transition_count")
        ),
        "teacher_checkpoint_sha256": manifest.get("teacher_checkpoint_sha256"),
        "slots": slots,
        "valid_window_count": valid_window_count,
        "observation_dim": observation_dim,
        "observation_layout": dict(layout),
        "chunk_length": chunk_length,
        "num_action_classes": action_classes,
        "action_layout": dict(action_layout),
        "teacher_observation_recipe": teacher_observation_recipe,
        "array_paths": array_paths,
        "window_indices_path": str(window_path),
    }


@dataclass
class _ProbeShard:
    info: dict[str, Any]
    window_indices: np.ndarray
    arrays: dict[str, np.ndarray]


class ProbeCollectionDataset:
    """Ordered collection pool backed by read-only, lazy NumPy memmaps.

    Only observations, controls, window indices, and the validity arrays needed
    to recheck each selected window are opened. Teacher logits and collector
    bookkeeping arrays are never mapped.
    """

    def __init__(
        self,
        manifest_paths: Sequence[str | Path] | str | Path,
        *,
        expected_split: str = "train",
    ) -> None:
        if expected_split not in {"train", "validation", "test"}:
            raise ValueError("expected_split must be 'train', 'validation', or 'test'")
        if isinstance(manifest_paths, (str, Path)):
            selected_paths = [manifest_paths]
        else:
            selected_paths = list(manifest_paths)
        if not selected_paths:
            raise ValueError("manifest_paths must contain at least one manifest")

        self.expected_split = expected_split
        self._shards: list[_ProbeShard] = []
        self._memmaps: list[np.memmap] = []
        self._closed = False
        self.observation_dim: int | None = None
        self.chunk_length: int | None = None
        self.num_action_classes: int | None = None
        self.observation_layout: dict[str, Any] | None = None
        self.action_layout: dict[str, Any] | None = None
        self.teacher_observation_recipe: dict[str, Any] | None = None
        self._offsets = np.zeros((len(selected_paths) + 1,), dtype=np.int64)

        try:
            reference: dict[str, Any] | None = None
            windows_by_round: dict[int, int] = {}
            for selected_path in selected_paths:
                manifest_path = Path(selected_path).expanduser().resolve(strict=True)
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"streaming manifest is not a file: {manifest_path}")
                info = _inspect_manifest(manifest_path, expected_split=expected_split)
                if reference is None:
                    reference = info
                    self.observation_dim = info["observation_dim"]
                    self.chunk_length = info["chunk_length"]
                    self.num_action_classes = info["num_action_classes"]
                    self.observation_layout = info["observation_layout"]
                    self.action_layout = info["action_layout"]
                else:
                    self._check_compatible(reference, info)
                candidate_recipe = info["teacher_observation_recipe"]
                if candidate_recipe is not None:
                    if self.teacher_observation_recipe is None:
                        self.teacher_observation_recipe = candidate_recipe
                    elif _canonical_json(candidate_recipe) != _canonical_json(
                        self.teacher_observation_recipe
                    ):
                        raise ValueError(
                            "selected manifests have incompatible dt/dynamics/observation normalization settings: "
                            f"{info['manifest_path']}"
                        )

                count = info["valid_window_count"]
                windows_by_round[info["collection_round_idx"]] = (
                    windows_by_round.get(info["collection_round_idx"], 0) + count
                )
                window_indices = self._open_window_indices(info)
                arrays = self._open_shard_arrays(info)
                self._shards.append(
                    _ProbeShard(info=info, window_indices=window_indices, arrays=arrays)
                )
                self._offsets[len(self._shards)] = self._offsets[len(self._shards) - 1] + count

            if expected_split == "train" and any(count == 0 for count in windows_by_round.values()):
                empty_rounds = [round_idx for round_idx, count in windows_by_round.items() if count == 0]
                raise ValueError(f"training collection rounds have no valid windows: {empty_rounds}")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check_compatible(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
        for field in ("observation_dim", "chunk_length", "num_action_classes"):
            if candidate[field] != reference[field]:
                raise ValueError(
                    f"selected manifests disagree on {field}: {candidate['manifest_path']}"
                )
        for field in ("observation_layout", "action_layout"):
            if _canonical_json(candidate[field]) != _canonical_json(reference[field]):
                raise ValueError(
                    f"selected manifests have incompatible {field.replace('_', ' ')}: "
                    f"{candidate['manifest_path']}"
                )
    def _open_memmap(
        self,
        path: str,
        *,
        field: str,
        expected_shape: tuple[int, ...],
        expected_dtype: str,
    ) -> np.memmap:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(array, np.memmap):
            raise ValueError(f"collection array {field} did not open as a memmap: {path}")
        if array.shape != expected_shape or array.dtype != np.dtype(expected_dtype):
            array._mmap.close()
            raise ValueError(
                f"collection array {field} has shape/dtype {array.shape}/{array.dtype}, "
                f"expected {expected_shape}/{np.dtype(expected_dtype)}"
            )
        if array.flags.writeable:
            array._mmap.close()
            raise ValueError(f"collection array {field} is not read-only: {path}")
        self._memmaps.append(array)
        return array

    def _open_window_indices(self, info: Mapping[str, Any]) -> np.ndarray:
        count = info["valid_window_count"]
        if count == 0:
            # Empty NPY payloads cannot be memory-mapped on every platform.
            return np.empty((0,), dtype=np.int64)
        return self._open_memmap(
            info["window_indices_path"],
            field="window_indices",
            expected_shape=(count,),
            expected_dtype="int64",
        )

    def _open_shard_arrays(self, info: Mapping[str, Any]) -> dict[str, np.ndarray]:
        steps = info["simulator_steps"]
        slots = info["slots"]
        dimension = info["observation_dim"]
        expected = {
            "observations": ((steps + 1, slots, dimension), "float32"),
            "controls": ((steps, slots, 2), "float32"),
            "transition_valid": ((steps, slots), "bool"),
            "endpoint_valid": ((steps + 1, slots), "bool"),
            "generation": ((steps + 1, slots), "int64"),
        }
        return {
            name: self._open_memmap(
                info["array_paths"][name],
                field=name,
                expected_shape=shape,
                expected_dtype=dtype,
            )
            for name, (shape, dtype) in expected.items()
        }

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @staticmethod
    def _normalize_indices(indices: Any, dataset_length: int) -> np.ndarray:
        if hasattr(indices, "detach") and hasattr(indices, "cpu"):
            indices = indices.detach().cpu().numpy()
        raw_indices = np.asarray(indices)
        if raw_indices.ndim == 0:
            raw_indices = raw_indices.reshape(1)
        if raw_indices.ndim != 1:
            raise TypeError("batch indices must be a one-dimensional integer array")
        if raw_indices.size == 0:
            return np.empty((0,), dtype=np.int64)
        if not np.issubdtype(raw_indices.dtype, np.integer) or np.issubdtype(
            raw_indices.dtype, np.bool_
        ):
            raise TypeError("batch indices must be a one-dimensional integer array")
        normalized = raw_indices.astype(np.int64, copy=False)
        if np.any(normalized < 0) or np.any(normalized >= dataset_length):
            raise IndexError(f"batch index is out of range for {dataset_length} windows")
        return normalized

    def get_batch(self, indices: Any) -> "ProbeBatch":
        """Fetch requested windows in caller order without reading full arrays."""

        if self._closed:
            raise RuntimeError("ProbeCollectionDataset is closed")
        normalized = self._normalize_indices(indices, len(self))
        assert self.observation_dim is not None
        assert self.chunk_length is not None
        batch_count = len(normalized)
        current = np.empty((batch_count, self.observation_dim), dtype=np.float32)
        future = np.empty((batch_count, self.observation_dim), dtype=np.float32)
        controls = np.empty((batch_count, self.chunk_length, 2), dtype=np.float32)
        source_ids: list[Any] = [None] * batch_count

        shard_indices = np.searchsorted(self._offsets[1:], normalized, side="right")
        for shard_idx in np.unique(shard_indices):
            batch_positions = np.flatnonzero(shard_indices == shard_idx)
            shard = self._shards[int(shard_idx)]
            local_indices = normalized[batch_positions] - self._offsets[int(shard_idx)]
            flattened_starts = np.asarray(shard.window_indices[local_indices], dtype=np.int64)
            slots = shard.info["slots"]
            steps = shard.info["simulator_steps"]
            if np.any(flattened_starts < 0) or np.any(flattened_starts >= steps * slots):
                raise ValueError(
                    f"manifest window_indices contains a flattened start outside [0, {steps * slots}): "
                    f"{shard.info['manifest_path']}"
                )
            transition_starts = flattened_starts // slots
            slot_indices = flattened_starts % slots
            horizon = self.chunk_length
            if np.any(transition_starts + horizon > steps):
                raise ValueError(
                    f"manifest window_indices contains a window truncated before t+K: "
                    f"{shard.info['manifest_path']}"
                )

            endpoint_indices = transition_starts[:, None] + np.arange(horizon + 1, dtype=np.int64)
            endpoint_valid = np.asarray(
                shard.arrays["endpoint_valid"][endpoint_indices, slot_indices[:, None]],
                dtype=np.bool_,
            )
            generations = np.asarray(
                shard.arrays["generation"][endpoint_indices, slot_indices[:, None]],
                dtype=np.int64,
            )
            transition_indices = transition_starts[:, None] + np.arange(horizon, dtype=np.int64)
            transition_valid = np.asarray(
                shard.arrays["transition_valid"][transition_indices, slot_indices[:, None]],
                dtype=np.bool_,
            )
            if not endpoint_valid.all():
                raise ValueError(
                    f"manifest window_indices selects an invalid observation endpoint: "
                    f"{shard.info['manifest_path']}"
                )
            if not transition_valid.all():
                raise ValueError(
                    f"manifest window_indices selects an invalid transition: "
                    f"{shard.info['manifest_path']}"
                )
            if np.any(generations != generations[:, :1]):
                raise ValueError(
                    f"manifest window_indices crosses an agent generation boundary: "
                    f"{shard.info['manifest_path']}"
                )

            observations = shard.arrays["observations"]
            current_values = np.asarray(
                observations[transition_starts, slot_indices, :], dtype=np.float32
            )
            future_values = np.asarray(
                observations[transition_starts + horizon, slot_indices, :], dtype=np.float32
            )
            control_values = np.asarray(
                shard.arrays["controls"][transition_indices, slot_indices[:, None], :],
                dtype=np.float32,
            )
            if not (
                np.isfinite(current_values).all()
                and np.isfinite(future_values).all()
                and np.isfinite(control_values).all()
            ):
                raise ValueError(f"selected probe window contains non-finite values: {shard.info['manifest_path']}")
            current[batch_positions] = current_values
            future[batch_positions] = future_values
            controls[batch_positions] = control_values
            from .contracts import WindowSource

            for output_idx, flattened_start in zip(batch_positions, flattened_starts):
                source_ids[int(output_idx)] = WindowSource(
                    manifest_path=shard.info["manifest_path"],
                    window_index=int(flattened_start),
                )

        import torch

        from .contracts import ProbeBatch

        return ProbeBatch(
            current_observations=torch.from_numpy(current),
            future_observations=torch.from_numpy(future),
            executed_controls=torch.from_numpy(controls),
            source_ids=tuple(source_ids),
        )

    def close(self) -> None:
        if self._closed:
            return
        seen: set[int] = set()
        for array in reversed(self._memmaps):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and id(mapping) not in seen:
                mapping.close()
                seen.add(id(mapping))
        self._memmaps.clear()
        self._shards.clear()
        self._closed = True

    def __enter__(self) -> "ProbeCollectionDataset":
        if self._closed:
            raise RuntimeError("ProbeCollectionDataset is closed")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class _SelectedIndices(Sequence[int]):
    """Deterministically shuffled indices with memory bounded by one block."""

    def __init__(
        self,
        dataset_length: int,
        *,
        seed: int,
        collection_index: int,
        epoch_index: int,
        max_windows: int | None,
    ) -> None:
        self.dataset_length = dataset_length
        self.length = min(dataset_length, max_windows) if max_windows is not None else dataset_length
        self.seed = seed
        self.collection_index = collection_index
        self.epoch_index = epoch_index
        self.full_block_count = dataset_length // _SHUFFLE_BLOCK_SIZE
        self.remainder = dataset_length % _SHUFFLE_BLOCK_SIZE
        block_rng = np.random.default_rng(
            np.random.SeedSequence([seed, collection_index, epoch_index, 0x5A17])
        )
        self.block_order = block_rng.permutation(self.full_block_count).astype(np.int64, copy=False)

    def __len__(self) -> int:
        return self.length

    def _block_values(self, block_idx: int) -> np.ndarray:
        block_start = block_idx * _SHUFFLE_BLOCK_SIZE
        block_length = min(_SHUFFLE_BLOCK_SIZE, self.dataset_length - block_start)
        block_rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self.collection_index, self.epoch_index, block_idx, 0xB10C]
            )
        )
        return (block_start + block_rng.permutation(block_length)).astype(np.int64, copy=False)

    def __getitem__(self, index: int | slice) -> int | np.ndarray:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            if step != 1:
                return np.fromiter(
                    (self[position] for position in range(start, stop, step)),
                    dtype=np.int64,
                )
            if start >= stop:
                return np.empty((0,), dtype=np.int64)
            chunks: list[np.ndarray] = []
            cursor = start
            full_shuffled_length = self.full_block_count * _SHUFFLE_BLOCK_SIZE
            while cursor < stop and cursor < full_shuffled_length:
                shuffled_block_position = cursor // _SHUFFLE_BLOCK_SIZE
                block_offset = cursor % _SHUFFLE_BLOCK_SIZE
                end = min(stop, (shuffled_block_position + 1) * _SHUFFLE_BLOCK_SIZE)
                physical_block = int(self.block_order[shuffled_block_position])
                chunks.append(self._block_values(physical_block)[block_offset : block_offset + end - cursor])
                cursor = end
            if cursor < stop:
                partial_block = self._block_values(self.full_block_count)
                partial_offset = cursor - full_shuffled_length
                chunks.append(partial_block[partial_offset : partial_offset + stop - cursor])
            if len(chunks) == 1:
                return chunks[0]
            return np.concatenate(chunks)

        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("selected index must be an integer or slice")
        index_int = int(index)
        if index_int < 0:
            index_int += self.length
        if index_int < 0 or index_int >= self.length:
            raise IndexError(f"selected index {index} is out of range for {self.length} windows")
        return int(self[index_int : index_int + 1][0])


def selected_indices(
    dataset_length: int,
    *,
    seed: int,
    epoch_index: int,
    max_windows: int | None = None,
    collection_index: int = 0,
) -> Sequence[int]:
    """Return deterministic shuffled indices without allocating an N-sized permutation."""

    for value, name in (
        (dataset_length, "dataset_length"),
        (seed, "seed"),
        (epoch_index, "epoch_index"),
        (collection_index, "collection_index"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if max_windows is not None and (
        isinstance(max_windows, bool)
        or not isinstance(max_windows, (int, np.integer))
        or int(max_windows) <= 0
    ):
        raise ValueError("max_windows must be a positive integer when specified")
    return _SelectedIndices(
        int(dataset_length),
        seed=int(seed),
        collection_index=int(collection_index),
        epoch_index=int(epoch_index),
        max_windows=None if max_windows is None else int(max_windows),
    )


def preflight_probe_data(config: Mapping[str, Any]) -> dict[str, Any]:
    """Audit selected train manifests and NPY headers without reading array values."""

    data = config["data"]
    collection_root = resolve_path(data["collection_root"]).resolve(strict=True)
    if not collection_root.is_dir():
        raise NotADirectoryError(f"collection_root is not a directory: {collection_root}")

    selected_rounds = list(data["collection_rounds"])
    source_ranks = list(data["source_ranks"])
    inspected: list[dict[str, Any]] = []
    per_round: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    teacher_observation_recipe: dict[str, Any] | None = None
    manifests_with_recipe = 0
    for collection_round_idx in selected_rounds:
        round_manifests: list[dict[str, Any]] = []
        for source_rank in source_ranks:
            manifest_path = collection_root / f"rank_{source_rank:03d}" / "train" / f"round_{collection_round_idx:04d}" / "manifest.json"
            resolved_manifest = manifest_path.resolve(strict=True)
            try:
                resolved_manifest.relative_to(collection_root)
            except ValueError as exc:
                raise ValueError(f"selected manifest escapes collection_root: {manifest_path}") from exc
            manifest_info = _inspect_manifest(
                resolved_manifest, expected_round_idx=collection_round_idx
            )
            if reference is None:
                reference = manifest_info
            else:
                for field in ("observation_dim", "chunk_length", "num_action_classes"):
                    if manifest_info[field] != reference[field]:
                        raise ValueError(
                            f"selected manifests disagree on {field}: {manifest_info['manifest_path']}"
                        )
                if _canonical_json(manifest_info["observation_layout"]) != _canonical_json(
                    reference["observation_layout"]
                ):
                    raise ValueError(
                        f"selected manifests have incompatible observation layouts: {manifest_info['manifest_path']}"
                    )
                if _canonical_json(manifest_info["action_layout"]) != _canonical_json(
                    reference["action_layout"]
                ):
                    raise ValueError(
                        f"selected manifests have incompatible action layouts: {manifest_info['manifest_path']}"
                    )
            current_recipe = manifest_info["teacher_observation_recipe"]
            if current_recipe is not None:
                manifests_with_recipe += 1
                if teacher_observation_recipe is None:
                    teacher_observation_recipe = current_recipe
                elif _canonical_json(current_recipe) != _canonical_json(teacher_observation_recipe):
                    raise ValueError(
                        "selected manifests have incompatible dt/dynamics/observation normalization settings: "
                        f"{manifest_info['manifest_path']}"
                    )
            inspected.append(manifest_info)
            round_manifests.append(manifest_info)

        per_round.append(
            {
                "collection_round_idx": collection_round_idx,
                "pooled_valid_window_count": sum(
                    manifest["valid_window_count"] for manifest in round_manifests
                ),
                "rank_window_counts": {
                    f"rank_{source_rank:03d}": manifest["valid_window_count"]
                    for source_rank, manifest in zip(source_ranks, round_manifests)
                },
                "manifest_paths": [manifest["manifest_path"] for manifest in round_manifests],
            }
        )

    if reference is None:
        raise ValueError("no selected training manifests were inspected")
    return {
        "collection_root": str(collection_root),
        "selected_rounds": selected_rounds,
        "source_ranks": source_ranks,
        "selected_manifests": [manifest["manifest_path"] for manifest in inspected],
        "per_round": per_round,
        "compatibility": {
            "observation_dim": reference["observation_dim"],
            "observation_layout": reference["observation_layout"],
            "chunk_length": reference["chunk_length"],
            "num_action_classes": reference["num_action_classes"],
            "action_layout": reference["action_layout"],
            "teacher_observation_recipe": teacher_observation_recipe,
        },
        "observation_recipe_compatibility": (
            "matched_across_selected_manifests"
            if manifests_with_recipe == len(inspected)
            else "unverified_missing_manifest_metadata"
        ),
        "scenario_separation": "unverified",
        "array_audit": "NPY headers, declared shapes/dtypes, paths, and exact byte lengths checked; array values and window index contents not scanned",
    }


def inspect_checkpoint_metadata(
    checkpoint_path: str | Path, compatibility: Mapping[str, Any]
) -> dict[str, Any]:
    """Load Condition B state onto CPU and compare exported metadata only."""

    import torch

    path = resolve_path(checkpoint_path).resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is not a file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != "condition_b_v1":
        raise ValueError("Expected a Condition B checkpoint with format condition_b_v1")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("Condition B checkpoint is missing model_config metadata")
    model = model_config.get("model")
    checkpoint_layout = model_config.get("observation_layout")
    if not isinstance(model, Mapping) or not isinstance(checkpoint_layout, Mapping):
        raise ValueError("Condition B checkpoint has invalid model/layout metadata")

    for key in (
        "observation_dim",
        "ego_features",
        "goal_features",
        "partner_features",
        "lane_features",
        "boundary_features",
        "traffic_control_features",
        "obs_valid_count_features",
        "context_dim",
        "goal_dim",
        "num_reward_coefs",
        "obs_slots_partners_n",
        "obs_slots_lane_kept",
        "obs_slots_boundary_kept",
        "obs_slots_traffic_controls_n",
    ):
        if key not in checkpoint_layout or checkpoint_layout[key] != compatibility["observation_layout"].get(key):
            raise ValueError(f"checkpoint observation_layout is incompatible at {key}")

    if model.get("chunk_length") != compatibility["chunk_length"]:
        raise ValueError("checkpoint prediction horizon disagrees with collection chunk_length")
    if model.get("num_action_classes") != compatibility["num_action_classes"]:
        raise ValueError("checkpoint action class count disagrees with collections")

    checkpoint_actions = np.asarray(model_config.get("action_table"), dtype=np.float64)
    collection_actions = np.asarray(
        compatibility["action_layout"].get("normalized_controls"), dtype=np.float64
    )
    if checkpoint_actions.shape != collection_actions.shape or not np.allclose(
        checkpoint_actions, collection_actions, atol=1e-6, rtol=1e-6
    ):
        raise ValueError("checkpoint normalized action table disagrees with collections")
    checkpoint_physical_actions = model_config.get("action_table_physical")
    collection_physical_actions = compatibility["action_layout"].get("physical_controls")
    if checkpoint_physical_actions is not None and collection_physical_actions is not None:
        checkpoint_physical = np.asarray(checkpoint_physical_actions, dtype=np.float64)
        collection_physical = np.asarray(collection_physical_actions, dtype=np.float64)
        if checkpoint_physical.shape != collection_physical.shape or not np.allclose(
            checkpoint_physical, collection_physical, atol=1e-6, rtol=1e-6
        ):
            raise ValueError("checkpoint physical action table disagrees with collections")
    checkpoint_recipe = _teacher_observation_recipe(model_config.get("teacher_config"))
    collection_recipe = compatibility.get("teacher_observation_recipe")
    if checkpoint_recipe is not None and collection_recipe is not None:
        if _canonical_json(checkpoint_recipe) != _canonical_json(collection_recipe):
            raise ValueError(
                "checkpoint teacher metadata disagrees with collection dt/dynamics/observation normalization settings"
            )
        recipe_status = "matched"
    else:
        recipe_status = "unverified_missing_teacher_config_metadata"
    return {
        "path": str(path),
        "format": "condition_b_v1",
        "inspection_device": "cpu",
        "observation_dim": compatibility["observation_dim"],
        "chunk_length": compatibility["chunk_length"],
        "num_action_classes": compatibility["num_action_classes"],
        "metadata_compatible": True,
        "observation_recipe_compatibility": recipe_status,
        "model_state_validated": False,
    }


def iter_collection_batches(
    manifest_paths: Sequence[str | Path],
    *,
    batch_size: int,
    seed: int,
    epoch_index: int,
    start_batch_index: int = 0,
    max_windows: int | None = None,
    collection_index: int = 0,
) -> Iterator["ProbeBatch"]:
    """Yield deterministic pooled batches from the requested batch cursor onward."""

    _positive_int(batch_size, "batch_size")
    _nonnegative_int(start_batch_index, "start_batch_index")
    dataset = ProbeCollectionDataset(manifest_paths, expected_split="train")
    try:
        order = selected_indices(
            len(dataset),
            seed=seed,
            epoch_index=epoch_index,
            max_windows=max_windows,
            collection_index=collection_index,
        )
        batch_count = math.ceil(len(order) / batch_size)
        if start_batch_index > batch_count:
            raise ValueError(
                f"start_batch_index {start_batch_index} exceeds epoch batch count {batch_count}"
            )
        for batch_index in range(start_batch_index, batch_count):
            start = batch_index * batch_size
            end = min(start + batch_size, len(order))
            yield dataset.get_batch(order[start:end])
    finally:
        dataset.close()
