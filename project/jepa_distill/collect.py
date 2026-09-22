"""One bounded round of frozen-teacher trajectory collection."""

from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING, Union

import numpy as np
import torch

from .dataset import build_window_index, validate_manifest

if TYPE_CHECKING:
    from torch.nn import Module


PathLike = Union[str, Path]
CollectionConfig = Mapping[str, Any]
_SCHEMA_VERSION = 1
_REQUIRED_SPLITS = ("train", "validation", "test")


def _nested(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = config.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{section} must be a mapping")
    return value


def _json_safe(value: Any, path: str = "config") -> Any:
    """Convert metadata to JSON values while rejecting non-finite numbers."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(child, f"{path}.{key}") for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child, f"{path}[{index}]") for index, child in enumerate(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist(), path)
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} contains NaN/Inf")
        return float(value)
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    raise ValueError(f"{path} contains unsupported metadata type {type(value).__name__}")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _as_observation_array(value: Any, *, expected_slots: Optional[int] = None) -> np.ndarray:
    observations = np.asarray(value, dtype=np.float32)
    if observations.ndim == 1:
        observations = observations[None, :]
    if observations.ndim != 2 or observations.shape[0] < 1 or observations.shape[1] < 1:
        raise ValueError(f"environment observations must have shape [N,D], got {observations.shape}")
    if expected_slots is not None and observations.shape[0] != expected_slots:
        raise ValueError(
            f"environment returned {observations.shape[0]} slots, expected {expected_slots}"
        )
    if not np.isfinite(observations).all():
        raise ValueError("environment observations contain NaN/Inf")
    return np.ascontiguousarray(observations.copy())


def _as_bool_array(value: Any, slots: int, field: str, *, default: bool = False) -> np.ndarray:
    if value is None:
        return np.full(slots, default, dtype=bool)
    result = np.asarray(value, dtype=bool)
    if result.ndim == 0:
        result = result.reshape(1)
    result = result.reshape(-1)
    if result.shape != (slots,):
        raise ValueError(f"environment {field} must have shape [{slots}], got {result.shape}")
    return result.copy()


def _parse_reset_result(result: Any) -> np.ndarray:
    if isinstance(result, tuple) and len(result) >= 1:
        result = result[0]
    if isinstance(result, list) and len(result) == 2 and isinstance(result[1], (Mapping, list, tuple)):
        result = result[0]
    return _as_observation_array(result)


def _parse_step_result(
    result: Any, masks: Any = None
) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray, Any, Optional[np.ndarray]]:
    if not isinstance(result, tuple) or len(result) < 4:
        raise ValueError("environment.step must return at least (obs, reward, terminated, truncated)")
    observations = _as_observation_array(result[0])
    slots = observations.shape[0]
    terminated = _as_bool_array(result[2], slots, "terminated")
    truncated = _as_bool_array(result[3], slots, "truncated")
    info = result[4] if len(result) >= 5 else None
    returned_masks = None
    if masks is not None:
        returned_masks = _as_bool_array(masks, slots, "masks", default=True)
    return observations, result[1], terminated, truncated, info, returned_masks


def _step_environment(
    env: Any, actions: np.ndarray
) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray, Any, Optional[np.ndarray]]:
    """Step through PufferLib's send/recv pair when available.

    ``pufferlib.vector.step`` intentionally drops ``recv()``'s agent mask.
    Condition B needs that mask for the action's pre-movement eligibility, so
    use the lower-level pair for PufferLib vectors and retain all seven fields.
    """

    if callable(getattr(env, "send", None)) and callable(getattr(env, "recv", None)):
        env.send(actions)
        received = env.recv()
        if not isinstance(received, tuple) or len(received) < 7:
            raise ValueError("PufferLib env.recv must return (obs,reward,term,trunc,info,agent_ids,masks)")
        agent_ids = np.asarray(received[5]).reshape(-1)
        if not np.array_equal(agent_ids, np.arange(agent_ids.size, dtype=agent_ids.dtype)):
            raise ValueError("collector requires stable vector slot order; env.recv returned reordered agent_ids")
        return _parse_step_result(received[:5], received[6])
    result = env.step(actions)
    if isinstance(result, tuple) and len(result) >= 7:
        agent_ids = np.asarray(result[5]).reshape(-1)
        if not np.array_equal(agent_ids, np.arange(agent_ids.size, dtype=agent_ids.dtype)):
            raise ValueError("collector requires stable vector slot order; env.step returned reordered agent_ids")
        return _parse_step_result(result[:5], result[6])
    return _parse_step_result(result)


def _extract_logits(output: Any, expected_slots: int, expected_classes: int) -> torch.Tensor:
    """Find the categorical teacher logits in Drive or a small test policy."""

    if isinstance(output, Mapping):
        for key in ("logits", "action_logits", "policy_logits"):
            if key in output:
                return _extract_logits(output[key], expected_slots, expected_classes)
    if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
        return _extract_logits(output.logits, expected_slots, expected_classes)
    if isinstance(output, np.ndarray):
        logits = torch.as_tensor(output)
    elif isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (list, tuple)):
        for child in output:
            try:
                return _extract_logits(child, expected_slots, expected_classes)
            except (TypeError, ValueError):
                continue
        raise ValueError("teacher output contains no categorical logits")
    else:
        raise TypeError(f"unsupported teacher output type {type(output).__name__}")
    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits[:, 0, :]
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    if logits.ndim != 2 or logits.shape[0] != expected_slots or logits.shape[1] != expected_classes:
        raise ValueError(
            f"teacher logits must have shape [{expected_slots},{expected_classes}], got {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("teacher logits contain NaN/Inf")
    return logits.detach().to(dtype=torch.float32)


def _teacher_logits(teacher: "Module", observations: np.ndarray, classes: int) -> torch.Tensor:
    try:
        teacher_device = next(teacher.parameters()).device
    except (AttributeError, StopIteration):
        teacher_device = torch.device("cpu")
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=teacher_device)
    with torch.no_grad():
        if hasattr(teacher, "forward_eval"):
            output = teacher.forward_eval(observation_tensor)
        else:
            output = teacher(observation_tensor)
    return _extract_logits(output, observations.shape[0], classes)


def _action_table(teacher: "Module", config: Mapping[str, Any], classes: int) -> np.ndarray:
    conversion = getattr(teacher, "discrete_actions_to_continuous", None)
    if callable(conversion):
        try:
            try:
                device = next(teacher.parameters()).device
            except (AttributeError, StopIteration):
                device = torch.device("cpu")
            action_ids = torch.arange(classes, device=device, dtype=torch.long)
            converted = conversion(action_ids)
            table = np.asarray(
                converted.detach().cpu() if isinstance(converted, torch.Tensor) else converted,
                dtype=np.float32,
            )
            if table.shape == (classes, 2) and np.isfinite(table).all() and not np.any(table < -1.000001) and not np.any(table > 1.000001):
                return np.ascontiguousarray(table.copy())
        except (RuntimeError, TypeError, ValueError):
            pass
    candidates = [
        getattr(teacher, "action_table", None),
        _nested(config, "collection").get("normalized_action_table"),
        _nested(config, "collection").get("action_table"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        table = np.asarray(candidate.detach().cpu() if isinstance(candidate, torch.Tensor) else candidate, dtype=np.float32)
        if table.shape != (classes, 2):
            raise ValueError(f"teacher action table must have shape [{classes},2], got {table.shape}")
        if not np.isfinite(table).all() or np.any(table < -1.000001) or np.any(table > 1.000001):
            raise ValueError("teacher normalized action table must be finite and lie in [-1,1]")
        return np.ascontiguousarray(table.copy())
    raise ValueError("teacher must expose action_table for categorical-to-continuous controls")


def _physical_action_table(teacher: "Module", config: Mapping[str, Any], classes: int) -> Optional[np.ndarray]:
    candidates = [
        getattr(teacher, "action_table_physical", None),
        _nested(config, "collection").get("physical_action_table"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        table = np.asarray(candidate.detach().cpu() if isinstance(candidate, torch.Tensor) else candidate, dtype=np.float32)
        if table.shape != (classes, 2):
            raise ValueError(f"teacher physical action table must have shape [{classes},2], got {table.shape}")
        if not np.isfinite(table).all():
            raise ValueError("teacher physical action table contains NaN/Inf")
        return np.ascontiguousarray(table.copy())
    return None


def _controls_from_teacher(
    teacher: "Module",
    logits: torch.Tensor,
    action_selection: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("teacher softmax probabilities contain NaN/Inf")
    if action_selection == "sample":
        actions = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    elif action_selection in ("mean", "mode"):
        actions = torch.argmax(probabilities, dim=-1)
    else:
        raise ValueError("collection.action_selection must be 'sample', 'mean', or 'mode'")

    if action_selection == "mean" and callable(getattr(teacher, "discrete_probs_to_continuous_mean", None)):
        controls = teacher.discrete_probs_to_continuous_mean(probabilities)
    elif action_selection != "mean" and callable(getattr(teacher, "discrete_actions_to_continuous", None)):
        controls = teacher.discrete_actions_to_continuous(actions)
    else:
        table = _action_table(teacher, config, logits.shape[-1])
        if action_selection == "mean":
            controls = probabilities.detach().cpu().numpy() @ table
        else:
            controls = table[actions.detach().cpu().numpy()]
    controls_array = np.asarray(
        controls.detach().cpu() if isinstance(controls, torch.Tensor) else controls, dtype=np.float32
    )
    if controls_array.shape != (logits.shape[0], 2):
        raise ValueError(f"teacher controls must have shape [{logits.shape[0]},2], got {controls_array.shape}")
    if not np.isfinite(controls_array).all() or np.any(controls_array < -1.000001) or np.any(controls_array > 1.000001):
        raise ValueError("teacher controls must be finite normalized values in [-1,1]")
    return actions.detach().cpu().numpy().astype(np.int32, copy=False), np.ascontiguousarray(controls_array.copy())


def _env_is_continuous(env: Any) -> bool:
    action_type = getattr(env, "action_type", getattr(env, "_action_type", None))
    if isinstance(action_type, str):
        return action_type == "continuous"
    action_space = getattr(env, "single_action_space", getattr(env, "action_space", None))
    return action_space is None or hasattr(action_space, "low")


def _environment_actions(env: Any, action_ids: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if _env_is_continuous(env):
        return np.ascontiguousarray(controls.copy())
    action_space = getattr(env, "single_action_space", getattr(env, "action_space", None))
    action_shape = getattr(action_space, "shape", None)
    if action_shape == (1,):
        return np.ascontiguousarray(action_ids.reshape(-1, 1).copy())
    if action_shape not in (None, ()):
        return np.ascontiguousarray(action_ids.reshape(-1, *action_shape).copy())
    return np.ascontiguousarray(action_ids.copy())


def _reported_masks(env: Any, slots: int) -> tuple[np.ndarray, bool]:
    masks = getattr(env, "masks", None)
    if masks is None:
        buffers = getattr(env, "buf", None)
        if isinstance(buffers, Mapping):
            masks = buffers.get("masks")
    if masks is None:
        return np.ones(slots, dtype=bool), False
    return _as_bool_array(masks, slots, "masks", default=True), True


def _info_has_reset(info: Any) -> bool:
    if isinstance(info, Mapping):
        for key in ("reset", "reset_event", "auto_reset", "autoreset", "episode_start"):
            value = info.get(key)
            if isinstance(value, (bool, np.bool_)) and bool(value):
                return True
        return any(_info_has_reset(value) for value in info.values())
    if isinstance(info, (list, tuple)):
        return any(_info_has_reset(value) for value in info)
    return False


def _observation_layout(config: Mapping[str, Any], teacher: "Module", env: Any, dimension: int) -> dict[str, Any]:
    candidate = _nested(config, "collection").get("observation_layout")
    if candidate is None:
        candidate = getattr(teacher, "condition_b_observation_layout", None)
    if candidate is None:
        candidate = getattr(teacher, "observation_layout", None)
    if candidate is None:
        candidate = getattr(env, "observation_layout", None)
    layout = dict(_json_safe(candidate)) if isinstance(candidate, Mapping) else {}
    declared_dimension = layout.get("observation_dim", layout.get("dim"))
    if declared_dimension is not None and int(declared_dimension) != dimension:
        raise ValueError(
            f"teacher observation layout width {declared_dimension} disagrees with env width {dimension}"
        )
    layout["observation_dim"] = dimension
    layout.setdefault("dtype", "float32")
    return layout


def _teacher_checkpoint_identity(teacher: "Module", config: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Return a cached checkpoint path and SHA-256 without hashing per step."""

    cached_hash = getattr(teacher, "condition_b_checkpoint_sha256", None)
    cached_path = getattr(teacher, "condition_b_checkpoint_path", None)
    if cached_hash is not None:
        return str(cached_path) if cached_path is not None else None, str(cached_hash)
    teacher_section = _nested(config, "teacher")
    raw_path = teacher_section.get("checkpoint")
    if raw_path is None:
        return None, None
    checkpoint_path = Path(raw_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        return str(checkpoint_path), None
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        while True:
            chunk = checkpoint_file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    checksum = digest.hexdigest()
    try:
        setattr(teacher, "condition_b_checkpoint_path", str(checkpoint_path))
        setattr(teacher, "condition_b_checkpoint_sha256", checksum)
    except Exception:
        pass
    return str(checkpoint_path), checksum


def _initial_state(
    env: Any, config: Mapping[str, Any], split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = getattr(env, "_jepa_collector_state", None)
    saved_observations = getattr(env, "_jepa_observations", None)
    if isinstance(state, Mapping) and saved_observations is not None:
        observations = _as_observation_array(saved_observations)
        slots = observations.shape[0]
        generations = np.asarray(state.get("reset_generation", np.zeros(slots)), dtype=np.int64).reshape(-1)
        active = np.asarray(state.get("slot_active", np.ones(slots)), dtype=bool).reshape(-1)
        if generations.shape != (slots,) or active.shape != (slots,):
            raise ValueError("saved collector state has the wrong slot count")
        return observations, generations.copy(), active.copy(), np.ones(slots, dtype=bool)

    if saved_observations is not None:
        observations = _as_observation_array(saved_observations)
    elif getattr(env, "_jepa_reset_done", False) and hasattr(env, "observations"):
        observations = _as_observation_array(env.observations)
    else:
        collection = _nested(config, "collection")
        seed = collection.get("seed")
        if seed is None:
            split_seeds = collection.get("split_seeds")
            if isinstance(split_seeds, Mapping):
                if split not in split_seeds:
                    raise ValueError(f"collection.split {split!r} has no collection.split_seeds entry")
                seed = split_seeds[split]
            else:
                seed = _nested(config, "training").get("seed")
        try:
            reset_result = env.reset(seed=None if seed is None else int(seed))
        except TypeError:
            reset_result = env.reset()
        observations = _parse_reset_result(reset_result)
        setattr(env, "_jepa_reset_done", True)
    slots = observations.shape[0]
    generations = np.zeros(slots, dtype=np.int64)
    active = np.ones(slots, dtype=bool)
    return observations, generations, active, np.ones(slots, dtype=bool)


def _new_segment(
    observation: np.ndarray,
    slot_idx: int,
    generation: int,
    segment_idx: int,
    scene_instance_id: int,
) -> dict[str, Any]:
    return {
        "slot_idx": int(slot_idx),
        "reset_generation": int(generation),
        "segment_idx": int(segment_idx),
        "scene_instance_id": int(scene_instance_id),
        "observations": [np.asarray(observation, dtype=np.float32).copy()],
        "state_valid": [True],
        "state_timestep": [0],
        "state_reset_generation": [int(generation)],
        "executed_controls": [],
        "teacher_logits": [],
        "timestep": [],
        "reset_generation_per_transition": [],
        "terminated": [],
        "truncated": [],
        "eligibility_mask": [],
        "valid_transition": [],
    }


def _finalize_segment(segment: Optional[dict[str, Any]], segments: list[dict[str, Any]]) -> None:
    if segment is None or not segment["executed_controls"]:
        return
    segments.append(segment)


def _scene_instance_ids(env: Any, slots: int) -> np.ndarray:
    """Expand Drive map IDs to the stable flat vector-agent slot order."""

    raw_map_ids = getattr(env, "map_ids", None)
    offsets = getattr(env, "agent_offsets", None)
    if raw_map_ids is not None and offsets is not None:
        map_ids = np.asarray(raw_map_ids).reshape(-1)
        offsets_array = np.asarray(offsets, dtype=np.int64).reshape(-1)
        if offsets_array.shape == (map_ids.size + 1,) and int(offsets_array[-1]) == slots:
            result = np.empty(slots, dtype=np.int64)
            for env_idx, map_id in enumerate(map_ids):
                result[offsets_array[env_idx] : offsets_array[env_idx + 1]] = int(map_id)
            return result
    return np.arange(slots, dtype=np.int64)


def _directory_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _pack_shard(segments: list[dict[str, Any]], observation_dim: int, classes: int) -> dict[str, np.ndarray]:
    trajectory_offsets = [0]
    observation_offsets = [0]
    observations: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    timesteps: list[np.ndarray] = []
    state_timesteps: list[np.ndarray] = []
    transition_generations: list[np.ndarray] = []
    state_generations: list[np.ndarray] = []
    valid_transition: list[np.ndarray] = []
    state_valid: list[np.ndarray] = []
    terminated: list[np.ndarray] = []
    truncated: list[np.ndarray] = []
    eligibility_mask: list[np.ndarray] = []
    trajectory_ids: list[str] = []
    trajectory_slot_idx: list[int] = []
    trajectory_reset_generation: list[int] = []
    for segment in segments:
        segment_observations = np.asarray(segment["observations"], dtype=np.float32)
        segment_controls = np.asarray(segment["executed_controls"], dtype=np.float32)
        segment_logits = np.asarray(segment["teacher_logits"], dtype=np.float32)
        segment_timesteps = np.asarray(segment["timestep"], dtype=np.int64)
        segment_state_timesteps = np.asarray(segment["state_timestep"], dtype=np.int64)
        segment_generations = np.asarray(segment["reset_generation_per_transition"], dtype=np.int64)
        segment_state_generations = np.asarray(segment["state_reset_generation"], dtype=np.int64)
        segment_valid = np.asarray(segment["valid_transition"], dtype=bool)
        segment_states = np.asarray(segment["state_valid"], dtype=bool)
        segment_terminated = np.asarray(segment["terminated"], dtype=bool)
        segment_truncated = np.asarray(segment["truncated"], dtype=bool)
        segment_masks = np.asarray(segment["eligibility_mask"], dtype=bool)
        transition_count = len(segment_controls)
        if transition_count == 0:
            segment_controls = np.empty((0, 2), dtype=np.float32)
            segment_logits = np.empty((0, classes), dtype=np.float32)
            segment_timesteps = np.empty((0,), dtype=np.int64)
            segment_generations = np.empty((0,), dtype=np.int64)
            segment_valid = np.empty((0,), dtype=bool)
            segment_terminated = np.empty((0,), dtype=bool)
            segment_truncated = np.empty((0,), dtype=bool)
            segment_masks = np.empty((0,), dtype=bool)
        if segment_observations.shape != (transition_count + 1, observation_dim):
            raise ValueError("collector segment observations do not align with controls")
        if segment_logits.shape != (transition_count, classes):
            raise ValueError("collector segment logits do not align with controls")
        if segment_timesteps.shape != (transition_count,) or segment_generations.shape != (transition_count,):
            raise ValueError("collector segment identity arrays do not align with controls")
        if segment_state_timesteps.shape != (transition_count + 1,) or segment_state_generations.shape != (transition_count + 1,):
            raise ValueError("collector segment state identity arrays do not align with observations")
        observations.append(segment_observations)
        controls.append(segment_controls)
        logits.append(segment_logits)
        timesteps.append(segment_timesteps)
        state_timesteps.append(segment_state_timesteps)
        transition_generations.append(segment_generations)
        state_generations.append(segment_state_generations)
        valid_transition.append(segment_valid)
        state_valid.append(segment_states)
        terminated.append(segment_terminated)
        truncated.append(segment_truncated)
        eligibility_mask.append(segment_masks)
        trajectory_offsets.append(trajectory_offsets[-1] + transition_count)
        observation_offsets.append(observation_offsets[-1] + transition_count + 1)
        trajectory_ids.append(
            f"slot{int(segment['slot_idx']):06d}_generation{int(segment['reset_generation']):06d}_segment{int(segment['segment_idx']):06d}"
        )
        trajectory_slot_idx.append(int(segment["slot_idx"]))
        trajectory_reset_generation.append(int(segment["reset_generation"]))
    return {
        "observations": np.concatenate(observations, axis=0).astype(np.float32, copy=False),
        "executed_controls": np.concatenate(controls, axis=0).astype(np.float32, copy=False),
        "teacher_logits": np.concatenate(logits, axis=0).astype(np.float32, copy=False),
        "timestep": np.concatenate(timesteps, axis=0).astype(np.int64, copy=False),
        "state_timestep": np.concatenate(state_timesteps, axis=0).astype(np.int64, copy=False),
        "reset_generation": np.concatenate(transition_generations, axis=0).astype(np.int64, copy=False),
        "state_reset_generation": np.concatenate(state_generations, axis=0).astype(np.int64, copy=False),
        "trajectory_offsets": np.asarray(trajectory_offsets, dtype=np.int64),
        "observation_offsets": np.asarray(observation_offsets, dtype=np.int64),
        "valid_transition": np.concatenate(valid_transition, axis=0).astype(bool, copy=False),
        "state_valid": np.concatenate(state_valid, axis=0).astype(bool, copy=False),
        "terminated": np.concatenate(terminated, axis=0).astype(bool, copy=False),
        "truncated": np.concatenate(truncated, axis=0).astype(bool, copy=False),
        "eligibility_mask": np.concatenate(eligibility_mask, axis=0).astype(bool, copy=False),
        "trajectory_ids": np.asarray(trajectory_ids, dtype="U128"),
        "trajectory_slot_idx": np.asarray(trajectory_slot_idx, dtype=np.int64),
        "trajectory_reset_generation": np.asarray(trajectory_reset_generation, dtype=np.int64),
        "trajectory_scene_instance_id": np.asarray(
            [int(segment["scene_instance_id"]) for segment in segments], dtype=np.int64
        ),
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output_file:
            np.savez_compressed(output_file, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path.stat().st_size


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _existing_round_transition_count(output_dir: Path) -> int:
    total = 0
    if not output_dir.exists():
        return 0
    for manifest_path in output_dir.glob("round_*/manifest.json"):
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if not isinstance(manifest, Mapping):
            raise ValueError(f"stored collection manifest is not a mapping: {manifest_path}")
        count = manifest.get("collection_transition_count", manifest.get("transition_count"))
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) < 0:
            raise ValueError(f"stored collection manifest has invalid transition count: {manifest_path}")
        total += int(count)
    return total


def collect_dataset(
    config: CollectionConfig,
    output_dir: PathLike,
    *,
    teacher: "Module",
    env: Any,
    collection_round_idx: int,
) -> Mapping[str, Any]:
    """Collect one bounded fresh round from a caller-owned teacher and env.

    The env is reset only on the first call when no ``_jepa_observations`` or
    ``_jepa_collector_state`` exists. Subsequent calls continue the live vector
    slots and start new round-local trajectories at their current observations.
    ``env.step`` outcomes belong to the action sent immediately before them;
    endpoint validity is proved with term/truncation/reset events and finite
    copied observations, never by treating the returned C mask as an endpoint
    mask. The returned mapping is JSON-compatible except for the explicit
    absolute ``manifest_path`` string and contains collection statistics.
    """

    if isinstance(collection_round_idx, bool) or int(collection_round_idx) < 0:
        raise ValueError("collection_round_idx must be a non-negative integer")
    collection_round_idx = int(collection_round_idx)
    collection = _nested(config, "collection")
    model = _nested(config, "model")
    chunk_length = _positive_int(model.get("chunk_length", collection.get("chunk_length", 4)), "model.chunk_length")
    classes = _positive_int(model.get("num_action_classes", 12), "model.num_action_classes")
    requested_transitions = _positive_int(
        collection.get("transitions_per_round"), "collection.transitions_per_round"
    )
    shard_transition_count = _positive_int(
        collection.get("shard_transition_count", 65536), "collection.shard_transition_count"
    )
    split = collection.get("split", "train")
    if not isinstance(split, str) or not split:
        raise ValueError("collection.split must be a non-empty string")
    if split not in _REQUIRED_SPLITS:
        raise ValueError(f"collection.split must be one of {_REQUIRED_SPLITS}, got {split!r}")
    action_selection = collection.get("action_selection", "sample")
    if action_selection not in ("sample", "mean", "mode"):
        raise ValueError("collection.action_selection must be 'sample', 'mean', or 'mode'")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    disk_root = output_root.parent if output_root.name in _REQUIRED_SPLITS else output_root
    round_dir = output_root / f"round_{collection_round_idx:04d}"
    if round_dir.exists():
        raise FileExistsError(f"collection round output already exists: {round_dir}")
    max_disk_bytes = collection.get("max_disk_bytes")
    if max_disk_bytes is not None:
        max_disk_bytes = _positive_int(max_disk_bytes, "collection.max_disk_bytes")
        if _directory_size(disk_root) > max_disk_bytes:
            raise RuntimeError("collection.max_disk_bytes is already exceeded before collection")

    observations, generations, active, _ = _initial_state(env, config, split)
    slots, observation_dim = observations.shape
    env_num_agents = getattr(env, "num_agents", slots)
    if int(env_num_agents) != slots:
        raise ValueError(f"env.num_agents={env_num_agents} disagrees with observations slots={slots}")
    remaining_transitions = collection.get("max_transitions")
    if remaining_transitions is not None:
        remaining_transitions = _positive_int(remaining_transitions, "collection.max_transitions") - _existing_round_transition_count(output_root)
        requested_transitions = min(requested_transitions, max(0, remaining_transitions))
        requested_transitions = (requested_transitions // slots) * slots
        if requested_transitions <= 0:
            raise RuntimeError("collection.max_transitions has no complete vector step remaining")
    if requested_transitions % slots:
        raise ValueError(
            "collection.transitions_per_round must be a multiple of env.num_agents "
            "so every vector slot has the same number of outcomes"
        )
    simulator_steps = requested_transitions // slots
    if simulator_steps < 1:
        raise ValueError("collection.transitions_per_round must include at least one vector step")

    if max_disk_bytes is not None:
        # This conservative lower bound is checked before any simulator step.
        # Compression can only reduce the eventual size; rejecting here avoids
        # spending a round of simulation that cannot fit the configured cap.
        minimum_shard_bytes = (
            (requested_transitions + slots) * observation_dim * np.dtype(np.float32).itemsize
            + requested_transitions * 2 * np.dtype(np.float32).itemsize
            + requested_transitions * classes * np.dtype(np.float32).itemsize
            + requested_transitions * 4
            + 8192
        )
        projected_lower_bound = _directory_size(disk_root) + minimum_shard_bytes
        if projected_lower_bound > max_disk_bytes:
            raise RuntimeError(
                "collection.max_disk_bytes is too small for the requested round "
                f"(minimum projected {projected_lower_bound} > {max_disk_bytes})"
            )

    layout = _observation_layout(config, teacher, env, observation_dim)
    segments: list[dict[str, Any]] = []
    scene_instance_ids = _scene_instance_ids(env, slots)
    current_segments: list[Optional[dict[str, Any]]] = [
        _new_segment(
            observations[slot_idx],
            slot_idx,
            int(generations[slot_idx]),
            0,
            int(scene_instance_ids[slot_idx]),
        )
        if active[slot_idx]
        else None
        for slot_idx in range(slots)
    ]
    segment_counts = np.zeros(slots, dtype=np.int64)
    rejected_reasons: dict[str, int] = {}
    valid_transition_count = 0
    stored_transition_count = 0
    mask_available = False
    started_at = time.monotonic()

    for _ in range(simulator_steps):
        logits = _teacher_logits(teacher, observations, classes)
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
            raise ValueError("environment observation shape changed within a collection round")
        if returned_masks is None:
            reported_masks, has_masks = _reported_masks(env, slots)
        else:
            reported_masks, has_masks = returned_masks, True
        mask_available = mask_available or has_masks
        reset_event = _info_has_reset(info)
        for slot_idx in range(slots):
            segment = current_segments[slot_idx]
            current_finite = bool(np.isfinite(observations[slot_idx]).all())
            endpoint_finite = bool(np.isfinite(next_observations[slot_idx]).all())
            boundary = bool(terminated[slot_idx] or truncated[slot_idx])
            pre_action_eligible = bool(reported_masks[slot_idx])
            valid = bool(
                segment is not None
                and pre_action_eligible
                and not boundary
                and current_finite
                and endpoint_finite
            )
            if not valid:
                reasons = []
                if segment is None:
                    reasons.append("inactive_slot")
                if not pre_action_eligible:
                    reasons.append("ineligible_mask")
                if terminated[slot_idx]:
                    reasons.append("terminated")
                if truncated[slot_idx]:
                    reasons.append("truncated")
                if not current_finite or not endpoint_finite:
                    reasons.append("nonfinite_observation")
                for reason in reasons or ["invalid_transition"]:
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            else:
                valid_transition_count += 1

            if segment is not None:
                transition_timestep = len(segment["executed_controls"])
                segment["executed_controls"].append(controls[slot_idx].copy())
                segment["teacher_logits"].append(logits[slot_idx].cpu().numpy().astype(np.float32, copy=True))
                segment["timestep"].append(transition_timestep)
                segment["reset_generation_per_transition"].append(int(segment["reset_generation"]))
                segment["terminated"].append(bool(terminated[slot_idx]))
                segment["truncated"].append(bool(truncated[slot_idx]))
                segment["eligibility_mask"].append(pre_action_eligible)
                segment["valid_transition"].append(valid)
                segment["observations"].append(next_observations[slot_idx].copy())
                segment["state_valid"].append(valid)
                segment["state_timestep"].append(transition_timestep + 1)
                segment["state_reset_generation"].append(int(segment["reset_generation"]))
                stored_transition_count += 1

            should_start_new = bool(truncated[slot_idx] or (terminated[slot_idx] and reset_event))
            if boundary or not pre_action_eligible or segment is None:
                _finalize_segment(segment, segments)
                current_segments[slot_idx] = None
                active[slot_idx] = False
                if should_start_new:
                    generations[slot_idx] += 1
                    segment_counts[slot_idx] += 1
                    current_segments[slot_idx] = _new_segment(
                        next_observations[slot_idx],
                        slot_idx,
                        int(generations[slot_idx]),
                        int(segment_counts[slot_idx]),
                        int(scene_instance_ids[slot_idx]),
                    )
                    active[slot_idx] = True
            elif segment is not None:
                active[slot_idx] = True
            observations = next_observations

    for segment in current_segments:
        _finalize_segment(segment, segments)
    observations = np.ascontiguousarray(observations.copy())
    setattr(env, "_jepa_observations", observations.copy())
    setattr(
        env,
        "_jepa_collector_state",
        {
            "reset_generation": generations.astype(np.int64).tolist(),
            "slot_active": active.astype(bool).tolist(),
        },
    )

    # A segment with a terminal/truncation marker remains useful for provenance,
    # but its event transition and endpoint are excluded by build_window_index.
    segments_to_write = [segment for segment in segments if segment["executed_controls"]]
    round_dir.mkdir(parents=True, exist_ok=False)
    shard_dir = round_dir / "shards"
    shard_specs: list[dict[str, Any]] = []
    shard_segment_groups: list[list[dict[str, Any]]] = []
    previous_segments: list[dict[str, Any]] = []
    current_shard_idx = 0
    current_transition_count = 0
    stored_shard_arrays: list[tuple[Path, dict[str, np.ndarray]]] = []

    def flush_shard(shard_segments: list[dict[str, Any]], shard_idx: int) -> None:
        if not shard_segments:
            return
        arrays = _pack_shard(shard_segments, observation_dim, classes)
        path = shard_dir / f"shard_{shard_idx:06d}.npz"
        shard_specs.append(
            {
                "path": str(path.relative_to(round_dir)),
                "trajectory_count": len(shard_segments),
                "transition_count": int(arrays["executed_controls"].shape[0]),
                "observation_dim": observation_dim,
                "chunk_length": chunk_length,
                "num_action_classes": classes,
            }
        )
        stored_shard_arrays.append((path, arrays))
        shard_segment_groups.append(list(shard_segments))

    for segment in segments_to_write:
        transition_count = len(segment["executed_controls"])
        if previous_segments and current_transition_count + transition_count > shard_transition_count:
            flush_shard(previous_segments, current_shard_idx)
            current_shard_idx += 1
            previous_segments = []
            current_transition_count = 0
        previous_segments.append(segment)
        current_transition_count += transition_count
    flush_shard(previous_segments, current_shard_idx)
    if not shard_specs:
        empty_segment = _new_segment(np.zeros(observation_dim, dtype=np.float32), 0, 0, 0, 0)
        empty_segment["executed_controls"] = []
        empty_segment["teacher_logits"] = []
        empty_segment["terminated"] = []
        empty_segment["truncated"] = []
        empty_segment["eligibility_mask"] = []
        empty_segment["valid_transition"] = []
        empty_segment["observations"] = [np.zeros(observation_dim, dtype=np.float32)]
        empty_segment["state_valid"] = [False]
        flush_shard([empty_segment], 0)

    if max_disk_bytes is not None:
        estimated_round_bytes = sum(
            sum(array.nbytes for array in arrays.values()) + 4096
            for _path, arrays in stored_shard_arrays
        ) + 8192
        projected = _directory_size(disk_root) + estimated_round_bytes
        if projected > max_disk_bytes:
            raise RuntimeError(
                "collection.max_disk_bytes would be exceeded before writing shards: "
                f"projected at least {projected} > {max_disk_bytes}"
            )

    for shard_idx, (path, arrays) in enumerate(stored_shard_arrays):
        _write_npz(path, arrays)

    split_refs: dict[str, list[dict[str, int]]] = {
        split_name: [] for split_name in _REQUIRED_SPLITS
    }
    split_refs.setdefault(split, [])
    for shard_idx, shard_segments in enumerate(shard_segment_groups):
        for trajectory_idx, _segment in enumerate(shard_segments):
            split_refs[split].append(
                {"shard_idx": int(shard_idx), "trajectory_idx": int(trajectory_idx)}
            )
    total_trajectory_count = sum(spec["trajectory_count"] for spec in shard_specs)
    total_transition_count = sum(spec["transition_count"] for spec in shard_specs)
    split_metadata = {
        split_name: {
            "trajectory_refs": split_refs.get(split_name, []),
            "trajectory_count": len(split_refs.get(split_name, [])),
            "window_count": 0,
        }
        for split_name in _REQUIRED_SPLITS
    }
    checkpoint_path, checkpoint_sha256 = _teacher_checkpoint_identity(teacher, config)
    physical_action_table = _physical_action_table(teacher, config, classes)
    action_layout: dict[str, Any] = {
        "normalized_controls": _action_table(teacher, config, classes).tolist(),
        "control_order": ["longitudinal", "lateral"],
        "normalized_range": [-1.0, 1.0],
    }
    if physical_action_table is not None:
        action_layout["physical_controls"] = physical_action_table.tolist()
    split_seed = collection.get("split_seeds", {}).get(split) if isinstance(collection.get("split_seeds"), Mapping) else None
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset_root": str(round_dir),
        "disk_budget_root": str(disk_root),
        "collection_round_idx": collection_round_idx,
        "split": split,
        "collection_mode": "teacher_driven",
        "action_selection": action_selection,
        "chunk_length": chunk_length,
        "num_action_classes": classes,
        "observation_layout": layout,
        "action_layout": action_layout,
        "teacher_checkpoint": checkpoint_path,
        "teacher_checkpoint_sha256": checkpoint_sha256,
        "split_seed": split_seed,
        "effective_config": _json_safe(config),
        "code_revision": config.get("code_revision"),
        "code_revision": os.environ.get("GIT_COMMIT"),
        "shards": shard_specs,
        "splits": split_metadata,
        "split_metadata": split_metadata,
        "trajectory_count": total_trajectory_count,
        "transition_count": total_transition_count,
        "collection_transition_count": requested_transitions,
        "shard_transition_count": shard_transition_count,
        "eligibility_mask_timing": "action_pre_movement",
        "endpoint_validity": "finite_next_observation_and_lifecycle_events",
        "stats": {
            "requested_transition_count": requested_transitions,
            "collected_transition_count": requested_transitions,
            "simulator_steps": simulator_steps,
            "stored_transition_count": stored_transition_count,
            "valid_transition_count": valid_transition_count,
            "rejected_transition_count": requested_transitions - valid_transition_count,
            "valid_count": valid_transition_count,
            "rejected_count": requested_transitions - valid_transition_count,
            "rejected_reasons": rejected_reasons,
            "eligibility_mask_available": mask_available,
            "collection_seconds": time.monotonic() - started_at,
        },
    }
    validation_manifest = dict(manifest)
    validation_manifest["_dataset_root"] = str(round_dir)
    validate_manifest(validation_manifest)
    windows = build_window_index(validation_manifest, split=split)
    candidate_window_count = sum(
        max(0, len(segment["executed_controls"]) - chunk_length + 1)
        for segment in segments_to_write
    )
    manifest["splits"][split]["window_count"] = len(windows)
    manifest["split_metadata"][split]["window_count"] = len(windows)
    manifest["valid_window_count"] = len(windows)
    manifest["candidate_window_count"] = candidate_window_count
    manifest["rejected_window_count"] = candidate_window_count - len(windows)
    manifest["stats"]["valid_window_count"] = len(windows)
    manifest["stats"]["candidate_window_count"] = candidate_window_count
    manifest["stats"]["rejected_window_count"] = candidate_window_count - len(windows)
    manifest["cost"] = {
        "simulator_steps": simulator_steps,
        "agent_transitions": requested_transitions,
        "stored_transitions": stored_transition_count,
        "collection_seconds": manifest["stats"]["collection_seconds"],
        "storage_bytes_before_manifest": _directory_size(round_dir),
    }
    manifest_path = round_dir / "manifest.json"
    if max_disk_bytes is not None and _directory_size(disk_root) + 4096 > max_disk_bytes:
        raise RuntimeError("collection.max_disk_bytes would be exceeded before writing manifest.json")
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    if len(windows) == 0:
        raise RuntimeError(
            f"collection round {collection_round_idx} produced no valid K={chunk_length} windows; "
            "inspect stats.rejected_reasons and reset boundaries"
        )
    return manifest
