"""Verify binned action probes against their selected reference rows and arrays."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .initial_geometry import initial_geometry
from .probe_returns import compute_probe_measurements


ACTION_IDS = tuple(range(12))
CONTINUATION_CAP = 2560
AXIS_EDGES = {
    "nearest_observed_partner_center_distance_m": (0.0, 5.0, 10.0, 20.0, 40.0, 80.0, math.inf),
    "nearest_observed_road_boundary_distance_m": (0.0, 1.5, 2.0, 3.0, 5.0, 8.0, math.inf),
    "nearest_visible_goal_distance_m": (0.0, 2.0, 5.0, 12.0, 30.0, 100.0, 300.0, math.inf),
}
DISTANCE_FIELDS = (*AXIS_EDGES, "current_goal_distance_m")
SAFETY_EVENTS = ("collision", "at_fault_collision", "offroad", "red_light")
REQUIRED_EVAL_METRICS = frozenset({
    "n",
    "offroad_rate",
    "episode_length",
    "collision_rate",
    "episode_return",
    "red_light_violation_rate",
    "comfort_violation_count",
    "velocity_progress_sum",
    "num_goals_reached",
    "lane_center_rate",
    "dnf_rate",
    "score",
    "avg_speed_per_agent",
    "avg_distance_per_infraction",
    "reward_components/collision",
    "reward_components/offroad",
    "reward_components/red_light",
    "reward_components/goal",
    "reward_components/lane_align",
    "reward_components/lane_center",
    "reward_components/comfort",
    "reward_components/velocity",
    "reward_components/timestep",
    "reward_components/reverse",
    "reward_components/overspeed",
    "reward_components/ade",
    "at_fault_collision_rate",
    "puffer_score",
    "driving_direction_score",
    "speed_limit_compliance",
    "making_progress_rate",
    "progress_ratio",
    "comfort_score",
    "multi_lane_time",
    "multi_lane_score",
    "active_agent_count",
    "episode_timestep",
    "agents_per_batch",
    "total_distance_travelled",
    "total_infractions",
})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _same_scalar(expected, actual, *, rtol=1e-6, atol=1e-7) -> bool:
    if expected is None or actual is None:
        other = actual if expected is None else expected
        if other is None:
            return True
        try:
            return math.isnan(float(other))
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(expected, (bool, np.bool_)) or isinstance(actual, (bool, np.bool_)):
        return bool(expected) == bool(actual)
    try:
        expected_value = float(expected)
        actual_value = float(actual)
    except (TypeError, ValueError, OverflowError):
        return expected == actual
    if math.isnan(expected_value) or math.isnan(actual_value):
        return math.isnan(expected_value) and math.isnan(actual_value)
    return math.isclose(expected_value, actual_value, rel_tol=rtol, abs_tol=atol)


def _edge_pair(edges, bin_index):
    if bin_index < 0 or bin_index >= len(edges) - 1:
        raise ValueError(f"bin_index {bin_index} is outside {len(edges) - 1} bins")
    return float(edges[bin_index]), float(edges[bin_index + 1])


def _expected_bin_index(value, edges):
    value = float(value)
    if not math.isfinite(value) or value < edges[0]:
        return None
    for index, upper in enumerate(edges[1:]):
        if value < upper:
            return index
    return None


def _edge_value_matches(actual, expected):
    if math.isinf(expected):
        return actual is None or (isinstance(actual, (int, float)) and math.isinf(float(actual)))
    return actual is not None and _same_scalar(expected, actual, rtol=0.0, atol=1e-9)


def _reference_shard_path(source_shard, references_dir: Path, state: dict) -> Path:
    source_path = Path(str(source_shard)).expanduser()
    if source_path.is_absolute():
        candidates = [source_path]
    else:
        # Selector paths are relative to records/, so they normally begin with
        # "references/" rather than being relative to the references directory.
        candidates = [references_dir / source_path, references_dir.parent / source_path]
    if not source_path.suffix:
        candidates.extend(path.with_suffix(".npz") for path in tuple(candidates))
    episode = state.get("episode_index")
    map_name = state.get("map_name")
    if episode is not None and map_name is not None:
        candidates.append(references_dir / str(map_name) / f"episode_{int(episode):05d}.npz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot locate source reference shard {source_shard!r}")


def _verify_reference_state(state, references_dir: Path, geometry_signature, cache):
    state_id = str(state.get("state_id", "<missing-state-id>"))
    source_shard = state.get("source_shard")
    if not source_shard:
        raise ValueError(f"{state_id}: selection state has no source_shard")
    shard_path = _reference_shard_path(source_shard, references_dir, state)
    if shard_path not in cache:
        sidecar_path = shard_path.with_suffix(".json")
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"{state_id}: reference sidecar is missing: {sidecar_path}")
        sidecar = _read_json(sidecar_path)
        actual_hash = _sha256_file(shard_path)
        if sidecar.get("data_sha256") != actual_hash:
            raise ValueError(f"{state_id}: source reference shard hash differs from its JSON sidecar")
        if geometry_signature and sidecar.get("run_signature") != geometry_signature:
            raise ValueError(f"{state_id}: source shard run_signature differs from geometry manifest")
        state_metadata = state.get("source_metadata")
        if state_metadata:
            expected_sidecar = (references_dir.parent / str(state_metadata)).resolve()
            if expected_sidecar != sidecar_path.resolve():
                raise ValueError(f"{state_id}: source_metadata does not identify its shard sidecar")
        with np.load(shard_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        cache[shard_path] = (sidecar, arrays, actual_hash)
    sidecar, arrays, actual_hash = cache[shard_path]

    for name in ("frame_idx", "agent_id", "slot", "current_goal_idx", "goal_count", "eligible", *DISTANCE_FIELDS):
        if name not in arrays:
            raise ValueError(f"{state_id}: source reference shard lacks array {name!r}")
    row_mask = (
        (arrays["frame_idx"] == int(state["frame_idx"]))
        & (arrays["agent_id"] == int(state["agent_id"]))
        & (arrays["slot"] == int(state["slot"]))
    )
    row_indices = np.flatnonzero(row_mask)
    if row_indices.size != 1:
        raise ValueError(f"{state_id}: raw reference join found {row_indices.size} matching rows, expected one")
    row_index = int(row_indices[0])
    frame_idx = int(state["frame_idx"])
    slot = int(state["slot"])
    if frame_idx < 0 or frame_idx >= len(arrays["fingerprints"]):
        raise ValueError(f"{state_id}: frame_idx is outside the saved reference episode")
    if not bool(arrays["eligible"][row_index]):
        raise ValueError(f"{state_id}: selected raw row is not eligible")
    if not bool(arrays["preaction_eligible"][frame_idx, slot]):
        raise ValueError(f"{state_id}: selected slot is not pre-action eligible")
    for field in ("current_goal_idx", "goal_count"):
        if int(state[field]) != int(arrays[field][row_index]):
            raise ValueError(f"{state_id}: {field} differs from its raw reference row")
    distances = state.get("distances_m")
    if not isinstance(distances, dict):
        raise ValueError(f"{state_id}: distances_m must be a mapping")
    for field in DISTANCE_FIELDS:
        raw_value = float(arrays[field][row_index])
        selected_value = distances.get(field)
        if math.isnan(raw_value):
            if selected_value is not None:
                raise ValueError(f"{state_id}: {field} should be null for its missing raw distance")
        elif selected_value is None or not _same_scalar(raw_value, selected_value, rtol=0.0, atol=1e-5):
            raise ValueError(f"{state_id}: {field} differs from its raw reference row")
    for state_field, array_name in (
        ("fingerprint", "fingerprints"),
        ("observation_fingerprint", "observation_fingerprints"),
    ):
        if state.get(state_field) != str(arrays[array_name][frame_idx]):
            raise ValueError(f"{state_id}: {state_field} differs from its source shard")
    for field in ("map_index", "map_name", "episode_index"):
        if field in state and state[field] != sidecar.get(field):
            raise ValueError(f"{state_id}: {field} differs from its source shard sidecar")
    if state.get("map_path") and state["map_path"] != sidecar.get("map_path"):
        raise ValueError(f"{state_id}: map_path differs from its source shard sidecar")
    return str(shard_path), row_index, actual_hash


def _verify_axis_contract(selection, expected_per_bin):
    errors = []
    axes = selection.get("axes")
    if not isinstance(axes, dict):
        return ["selection.axes must be a mapping"]
    if set(axes) != set(AXIS_EDGES):
        errors.append(f"selection.axes keys differ from expected axes: {sorted(axes)}")
    for axis, expected_edges in AXIS_EDGES.items():
        spec = axes.get(axis)
        if isinstance(spec, dict):
            actual_edges = spec.get("edges_m", spec.get("edges", spec.get("bin_edges_m")))
        else:
            actual_edges = spec
        if actual_edges is None or len(actual_edges) != len(expected_edges):
            errors.append(f"selection axis {axis} does not expose the agreed bin edges")
            continue
        for index, (actual, expected) in enumerate(zip(actual_edges, expected_edges)):
            if not _edge_value_matches(actual, expected):
                errors.append(f"selection axis {axis} edge {index} differs from the agreed edges")
                break
    if selection.get("requested_per_bin") != expected_per_bin:
        errors.append(
            f"selection.requested_per_bin is {selection.get('requested_per_bin')!r}, expected {expected_per_bin}"
        )
    return errors


def _verify_selection_states(selection, expected_per_bin, references_dir, geometry_signature):
    errors = _verify_axis_contract(selection, expected_per_bin)
    states = selection.get("states")
    if not isinstance(states, list):
        return errors + ["selection.states must be a list"], {}, {}, 0
    expected_state_count = expected_per_bin * sum(len(edges) - 1 for edges in AXIS_EDGES.values())
    if selection.get("selected_state_count") != len(states):
        errors.append("selection.selected_state_count differs from the states array length")
    if selection.get("requested_state_count") != expected_state_count:
        errors.append(f"selection.requested_state_count differs from {expected_state_count}")
    if len(states) != expected_state_count:
        errors.append(f"selection has {len(states)} states, expected {expected_state_count}")
    state_by_id = {}
    source_by_state = {}
    counts = Counter()
    unique_rows = set()
    shard_cache = {}
    for state in states:
        state_id = str(state.get("state_id", ""))
        if not state_id:
            errors.append("selection contains a state without state_id")
            continue
        if state_id in state_by_id:
            errors.append(f"selection repeats state_id {state_id}")
        state_by_id[state_id] = state
        assigned = state.get("assigned_bin")
        if not isinstance(assigned, dict):
            errors.append(f"{state_id}: assigned_bin must be a mapping")
            continue
        axis = assigned.get("axis")
        if axis not in AXIS_EDGES:
            errors.append(f"{state_id}: assigned_bin axis is invalid: {axis!r}")
            continue
        try:
            bin_index = int(assigned["bin_index"])
            lower, upper = _edge_pair(AXIS_EDGES[axis], bin_index)
            expected_id = f"{axis}:{bin_index:02d}"
            if assigned.get("bin_id") != expected_id:
                errors.append(f"{state_id}: assigned_bin bin_id differs from {expected_id}")
            if not _edge_value_matches(assigned.get("lower_m"), lower):
                errors.append(f"{state_id}: assigned_bin lower_m differs from agreed edge")
            if not _edge_value_matches(assigned.get("upper_m"), upper):
                errors.append(f"{state_id}: assigned_bin upper_m differs from agreed edge")
            value = (state.get("distances_m") or {}).get(axis)
            if value is None or _expected_bin_index(value, AXIS_EDGES[axis]) != bin_index:
                errors.append(f"{state_id}: assigned distance does not belong to its bin")
            counts[(axis, bin_index)] += 1
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            errors.append(f"{state_id}: invalid assigned_bin: {error}")

        bin_tags = state.get("bin_tags")
        if not isinstance(bin_tags, dict) or set(bin_tags) != set(AXIS_EDGES):
            errors.append(f"{state_id}: bin_tags must contain exactly the three configured axes")
        else:
            for tag_axis, edges in AXIS_EDGES.items():
                tag = bin_tags.get(tag_axis)
                distance = (state.get("distances_m") or {}).get(tag_axis)
                if not isinstance(tag, dict):
                    errors.append(f"{state_id}: bin_tags[{tag_axis}] must be a mapping")
                    continue
                expected_index = None if distance is None else _expected_bin_index(distance, edges)
                if tag.get("bin_index") != expected_index:
                    errors.append(f"{state_id}: bin_tags[{tag_axis}] index disagrees with measured distance")
                    continue
                if expected_index is None:
                    if tag.get("bin_id") is not None:
                        errors.append(f"{state_id}: missing distance should have no {tag_axis} bin_id")
                    continue
                lower, upper = _edge_pair(edges, expected_index)
                if tag.get("bin_id") != f"{tag_axis}:{expected_index:02d}":
                    errors.append(f"{state_id}: bin_tags[{tag_axis}] bin_id is invalid")
                if not _edge_value_matches(tag.get("lower_m"), lower) or not _edge_value_matches(
                    tag.get("upper_m"), upper
                ):
                    errors.append(f"{state_id}: bin_tags[{tag_axis}] bounds disagree with agreed edges")

        try:
            shard_path, row_index, shard_hash = _verify_reference_state(
                state, references_dir, geometry_signature, shard_cache
            )
            row_identity = (
                shard_path,
                int(state["frame_idx"]),
                int(state["agent_id"]),
                int(state["slot"]),
            )
            if row_identity in unique_rows:
                errors.append(f"{state_id}: raw reference state is duplicated in selection")
            unique_rows.add(row_identity)
            source_by_state[state_id] = {
                "shard_path": shard_path,
                "row_index": row_index,
                "shard_sha256": shard_hash,
                "sidecar_path": str(Path(shard_path).with_suffix(".json")),
                "sidecar_sha256": _sha256_file(Path(shard_path).with_suffix(".json")),
            }
        except (OSError, KeyError, TypeError, ValueError, IndexError, zipfile.BadZipFile) as error:
            errors.append(f"{state_id}: source raw row check failed: {error}")

    for axis, edges in AXIS_EDGES.items():
        for bin_index in range(len(edges) - 1):
            count = counts[(axis, bin_index)]
            if count != expected_per_bin:
                errors.append(f"bin {axis}:{bin_index:02d} has {count} unique assigned states; expected {expected_per_bin}")
    _verify_bin_summary(selection, counts, expected_per_bin, errors)
    shortages = selection.get("shortages")
    if shortages != []:
        errors.append(f"selection.shortages must be empty for a complete selection; got {shortages!r}")
    return errors, state_by_id, dict(counts), len(unique_rows), source_by_state


def _verify_bin_summary(selection, counts, expected_per_bin, errors):
    summary = selection.get("bin_summary")
    if not isinstance(summary, dict):
        errors.append("selection.bin_summary must be a mapping")
        return
    if set(summary) != set(AXIS_EDGES):
        errors.append("selection.bin_summary axes differ from the configured axes")
        return
    for axis, edges in AXIS_EDGES.items():
        entries = summary.get(axis)
        if not isinstance(entries, list) or len(entries) != len(edges) - 1:
            errors.append(f"selection.bin_summary[{axis}] has the wrong number of bins")
            continue
        for bin_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"selection.bin_summary[{axis}][{bin_index}] is invalid")
                continue
            lower, upper = _edge_pair(edges, bin_index)
            if entry.get("bin_index") != bin_index or entry.get("bin_id") != f"{axis}:{bin_index:02d}":
                errors.append(f"selection.bin_summary[{axis}][{bin_index}] has invalid identity")
            if not _edge_value_matches(entry.get("lower_m"), lower) or not _edge_value_matches(
                entry.get("upper_m"), upper
            ):
                errors.append(f"selection.bin_summary[{axis}][{bin_index}] has invalid bounds")
            if entry.get("selected_count") != counts.get((axis, bin_index), 0):
                errors.append(f"selection.bin_summary[{axis}][{bin_index}] selected_count disagrees with states")
            if entry.get("selected_count") != expected_per_bin or entry.get("shortage_count") != 0:
                errors.append(f"selection.bin_summary[{axis}][{bin_index}] is short of requested states")
            if entry.get("candidate_count", 0) < entry.get("selected_count", 0):
                errors.append(f"selection.bin_summary[{axis}][{bin_index}] candidate_count is too small")


def _verify_probe_bundle(
    json_path: Path,
    selection_state: dict,
    expected_state_id: str,
    source_info: dict | None = None,
    geometry_signature: str | None = None,
    source_env_config: dict | None = None,
):
    errors = []
    metadata = _read_json(json_path)
    state_id = str(metadata.get("state_id", ""))
    if state_id != expected_state_id:
        errors.append(f"{json_path}: state_id is {state_id!r}, expected {expected_state_id!r}")
    if metadata.get("status") != "complete":
        errors.append(f"{json_path}: status is not complete")
    if metadata.get("probe_id") != expected_state_id:
        errors.append(f"{json_path}: probe_id differs from state_id")
    data_name = metadata.get("data_file", json_path.with_suffix(".npz").name)
    data_path = (json_path.parent / str(data_name)).resolve()
    if not data_path.is_file():
        return errors + [f"{json_path}: missing saved array bundle {data_path}"], None
    expected_hash = metadata.get("data_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        errors.append(f"{json_path}: data_sha256 is missing")
    elif _sha256_file(data_path) != expected_hash:
        errors.append(f"{json_path}: saved NPZ hash differs from JSON data_sha256")
    try:
        with np.load(data_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        return errors + [f"{json_path}: cannot read saved NPZ arrays: {error}"], None

    state_record = metadata.get("selection_state", metadata.get("state", metadata.get("selected_state")))
    if state_record is None:
        errors.append(f"{json_path}: embedded selection_state is missing")
    elif state_record != selection_state:
        errors.append(f"{json_path}: embedded selection state differs from selection.json")
    selected_bin = selection_state.get("assigned_bin") or {}
    recorded_bin = metadata.get("assigned_bin")
    if not isinstance(recorded_bin, dict) or recorded_bin != selected_bin:
        errors.append(f"{json_path}: assigned_bin differs from selection.json")
    for field in ("source_shard", "fingerprint", "observation_fingerprint"):
        if field in metadata and metadata[field] != selection_state.get(field):
            errors.append(f"{json_path}: {field} differs from selection.json")
    probe = metadata.get("probe") or {}
    for probe_field, state_field in (
        ("frame_idx", "frame_idx"), ("agent_id", "agent_id"), ("ego_idx", "slot"),
        ("active_slot", "slot"),
    ):
        if probe_field in probe and probe[probe_field] != selection_state.get(state_field):
            errors.append(f"{json_path}: probe {probe_field} differs from selection.json")
    for probe_field, state_field in (
        ("reference_fingerprint", "fingerprint"),
        ("reference_observation_fingerprint", "observation_fingerprint"),
    ):
        if probe.get(probe_field) != selection_state.get(state_field):
            errors.append(f"{json_path}: probe {probe_field} differs from selection.json")
    if "initial_ego_observation" not in arrays:
        errors.append(f"{json_path}: initial_ego_observation is missing from the NPZ")
    elif source_env_config is not None:
        observation = np.asarray(arrays["initial_ego_observation"], dtype=np.float32)
        if observation.ndim != 1 or not np.all(np.isfinite(observation)):
            errors.append(f"{json_path}: initial_ego_observation must be a finite one-dimensional array")
        else:
            try:
                reconstructed_geometry = initial_geometry(
                    observation,
                    source_env_config,
                    current_goal_idx=int(selection_state["current_goal_idx"]),
                    goal_count=int(selection_state["goal_count"]),
                )
                recorded_geometry = probe.get("initial_ego_geometry") or {}
                distances = selection_state.get("distances_m") or {}
                for field in DISTANCE_FIELDS:
                    if not _same_scalar(reconstructed_geometry[field], recorded_geometry.get(field)):
                        errors.append(f"{json_path}: initial geometry {field} differs from the saved observation")
                    if not _same_scalar(reconstructed_geometry[field], distances.get(field)):
                        errors.append(f"{json_path}: saved initial observation {field} differs from the selected raw row")
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"{json_path}: cannot reconstruct initial ego geometry: {error}")
    source = metadata.get("source") or {}
    if not source:
        errors.append(f"{json_path}: source provenance block is missing")
    else:
        if source.get("shard") != selection_state.get("source_shard"):
            errors.append(f"{json_path}: source.shard differs from selection.json")
        if geometry_signature and source.get("geometry_manifest_signature") != geometry_signature:
            errors.append(f"{json_path}: source geometry manifest signature differs from selection.json")
        if source_info and source.get("shard_sha256") != source_info.get("shard_sha256"):
            errors.append(f"{json_path}: source shard hash differs from the second-run raw shard")
        if source_info and source.get("sidecar_sha256") != source_info.get("sidecar_sha256"):
            errors.append(f"{json_path}: source sidecar hash differs from the second-run raw sidecar")
        if source_info and Path(str(source.get("sidecar", ""))).resolve() != Path(
            source_info["sidecar_path"]
        ).resolve():
            errors.append(f"{json_path}: source sidecar path differs from the second-run raw sidecar")
        recorded_source_row = metadata.get("source_row")
        if not isinstance(recorded_source_row, dict):
            errors.append(f"{json_path}: copied source_row is missing")
        else:
            source_geometry_row = source.get("source_geometry_row")
            if source_geometry_row != recorded_source_row:
                errors.append(f"{json_path}: source row copies disagree")
            for field in ("frame_idx", "slot", "agent_id", "current_goal_idx", "goal_count"):
                if recorded_source_row.get(field) != selection_state.get(field):
                    errors.append(f"{json_path}: source row {field} differs from selection.json")
            for field in ("fingerprint", "observation_fingerprint"):
                if recorded_source_row.get(field) != selection_state.get(field):
                    errors.append(f"{json_path}: source row {field} differs from selection.json")
            row_distances = recorded_source_row.get("distances_m") or {}
            selected_distances = selection_state.get("distances_m") or {}
            for field in DISTANCE_FIELDS:
                if not _same_scalar(row_distances.get(field), selected_distances.get(field)):
                    errors.append(f"{json_path}: source row distance {field} differs from selection.json")

    branches = metadata.get("branches")
    if not isinstance(branches, list) or len(branches) != len(ACTION_IDS):
        return errors + [f"{json_path}: expected 12 branch rows"], None
    action_ids = [branch.get("action_idx") for branch in branches]
    if any(not isinstance(action_id, int) or isinstance(action_id, bool) for action_id in action_ids) or set(
        action_ids
    ) != set(ACTION_IDS):
        errors.append(f"{json_path}: branch action_idx values are not exactly 0..11")
        return errors, None
    repeat_ids = {branch.get("repeat_idx") for branch in branches}
    if len(repeat_ids) != 1:
        errors.append(f"{json_path}: expected one shared repeat_idx, got {sorted(map(str, repeat_ids))}")
    elif next(iter(repeat_ids)) != 0:
        errors.append(f"{json_path}: the only repeat_idx must be 0")
    if "action_idx" not in arrays or not np.array_equal(np.sort(arrays["action_idx"]), ACTION_IDS):
        errors.append(f"{json_path}: NPZ action_idx is not exactly 0..11")
    if "repeat_idx" not in arrays or len(np.unique(arrays["repeat_idx"])) != 1:
        errors.append(f"{json_path}: NPZ repeat_idx does not contain one repeat")
    elif int(arrays["repeat_idx"][0]) != 0:
        errors.append(f"{json_path}: the only NPZ repeat_idx must be 0")
    branch_indices = {branch.get("action_idx"): index for index, branch in enumerate(branches)}
    for array_name in ("rewards", "log_probs", "valid", "terminals", "truncations", "next_soft_values"):
        if array_name not in arrays or arrays[array_name].ndim != 2 or arrays[array_name].shape[1] != 12:
            errors.append(f"{json_path}: {array_name} must have shape [time, 12]")
    required_arrays = {
        "rewards", "log_probs", "valid", "terminals", "truncations", "next_soft_values",
        "goal_events", "action_idx", "repeat_idx", "continuation_seed", "task_id", "q1", "q2",
    }
    required_arrays.update(f"event_{name}" for name in SAFETY_EVENTS)
    missing_arrays = sorted(required_arrays - set(arrays))
    if missing_arrays:
        errors.append(f"{json_path}: NPZ is missing arrays {missing_arrays}")
        return errors, None

    for array_name in ("goal_events", *(f"event_{name}" for name in SAFETY_EVENTS)):
        if arrays[array_name].shape != arrays["valid"].shape or arrays[array_name].dtype.kind != "b":
            errors.append(f"{json_path}: {array_name} shape/dtype differs from valid")
    for array_name in ("action_idx", "repeat_idx", "continuation_seed", "task_id", "q1", "q2"):
        if arrays[array_name].shape != (12,):
            errors.append(f"{json_path}: {array_name} must have shape [12]")
    for array_name in ("q1", "q2"):
        if arrays[array_name].dtype.kind not in "fiu" or np.any(~np.isfinite(arrays[array_name])):
            errors.append(f"{json_path}: {array_name} must contain finite numeric Q estimates")

    model = metadata.get("model") or {}
    for name in ("gamma", "alpha", "return_estimator"):
        if name not in model:
            errors.append(f"{json_path}: model metadata lacks {name}")
    collection = metadata.get("collection") or {}
    if collection.get("continuation_horizon") != CONTINUATION_CAP:
        errors.append(f"{json_path}: collection continuation_horizon is not {CONTINUATION_CAP}")
    if errors:
        return errors, None
    try:
        measurements = compute_probe_measurements(
            arrays["rewards"], arrays["log_probs"], arrays["valid"], arrays["terminals"],
            arrays["truncations"], arrays["next_soft_values"], gamma=float(model["gamma"]),
            alpha=float(model["alpha"]), return_estimator=model["return_estimator"],
            pql_horizon=int(model.get("pql_horizon", 128)),
            pql_lambda=float(model.get("pql_lambda", 0.95)),
        )
    except (ValueError, TypeError, RuntimeError) as error:
        return errors + [f"{json_path}: saved arrays cannot reconstruct measurements: {error}"], None

    horizon_by_action = {}
    for action_id in ACTION_IDS:
        branch = branches[branch_indices[action_id]]
        branch_idx = branch_indices[action_id]
        if arrays["action_idx"][branch_idx] != action_id:
            errors.append(f"{json_path}: branch/NPZ action order mismatch at action {action_id}")
        for field, array_name in (("task_id", "task_id"), ("continuation_seed", "continuation_seed"),
                                  ("repeat_idx", "repeat_idx")):
            if field in branch and str(branch[field]) != str(arrays[array_name][branch_idx]):
                errors.append(f"{json_path}: action {action_id} {field} differs from NPZ")
        for field in ("q1", "q2"):
            if field not in branch or not _same_scalar(branch[field], arrays[field][branch_idx]):
                errors.append(f"{json_path}: action {action_id} {field} differs from NPZ")
        if not _same_scalar(branch.get("q_min"), min(float(arrays["q1"][branch_idx]), float(arrays["q2"][branch_idx]))):
            errors.append(f"{json_path}: action {action_id} q_min differs from q1/q2")
        horizon = int(measurements["observed_horizon"][branch_idx])
        horizon_by_action[action_id] = horizon
        if horizon > CONTINUATION_CAP:
            errors.append(f"{json_path}: action {action_id} exceeds {CONTINUATION_CAP}-step continuation cap")
        for metric_name, values in measurements.items():
            if not isinstance(values, np.ndarray) or values.ndim != 1:
                continue
            if metric_name not in branch:
                errors.append(f"{json_path}: action {action_id} lacks reconstructed field {metric_name}")
                continue
            if not _same_scalar(branch[metric_name], values[branch_idx]):
                errors.append(f"{json_path}: action {action_id} {metric_name} differs from saved-array reconstruction")
        for event in SAFETY_EVENTS:
            event_array = arrays[f"event_{event}"][:, branch_idx]
            expected_event = bool(np.any(event_array & arrays["valid"][:, branch_idx]))
            if bool(branch.get(event)) != expected_event:
                errors.append(f"{json_path}: action {action_id} {event} differs from saved event flags")
        goal_flags = arrays["goal_events"][:, branch_idx] & arrays["valid"][:, branch_idx]
        if int(branch.get("goals_reached", -1)) != int(np.count_nonzero(goal_flags)):
            errors.append(f"{json_path}: action {action_id} goals_reached differs from saved event flags")
        if bool(branch.get("goal_reached")) != bool(np.any(goal_flags)):
            errors.append(f"{json_path}: action {action_id} goal_reached differs from saved event flags")
        _verify_ending_reason(json_path, branch, horizon, errors, action_id)

    seeds = np.asarray(arrays["continuation_seed"])
    if len(np.unique(seeds)) != 1:
        errors.append(f"{json_path}: action branches do not share one continuation seed")
    task_ids = np.asarray(arrays["task_id"]).astype(str)
    if len(set(task_ids.tolist())) != 12:
        errors.append(f"{json_path}: task_id values are not unique across action branches")
    _verify_eval_metrics(json_path, metadata, arrays, branches, branch_indices, errors)
    counts = {
        "branch_count": len(branches),
        "action_count": len(set(action_ids)),
        "repeat_count": len(repeat_ids),
        "max_observed_horizon": max(horizon_by_action.values()) if horizon_by_action else 0,
    }
    return errors, counts


def _verify_ending_reason(json_path, branch, horizon, errors, action_id):
    reason = branch.get("ending_reason")
    if reason not in {"terminal", "scene_boundary", "probe_horizon", "validity_gap", "truncated", "censored"}:
        errors.append(f"{json_path}: action {action_id} has invalid ending_reason {reason!r}")
        return
    terminal = bool(branch.get("terminal"))
    truncated = bool(branch.get("truncated"))
    validity_gap = bool(branch.get("validity_gap"))
    scene_boundary = bool(branch.get("scene_boundary"))
    probe_horizon_reached = bool(branch.get("probe_horizon_reached"))
    if terminal:
        expected_reason = "terminal"
    elif probe_horizon_reached:
        expected_reason = "probe_horizon"
    elif scene_boundary:
        expected_reason = "scene_boundary"
    elif validity_gap:
        expected_reason = "validity_gap"
    elif truncated:
        expected_reason = "truncated"
    else:
        expected_reason = "censored"
    if reason != expected_reason:
        errors.append(
            f"{json_path}: action {action_id} ending_reason {reason!r} disagrees with boundary flags; "
            f"expected {expected_reason!r}"
        )
    if reason == "terminal" and not terminal:
        errors.append(f"{json_path}: action {action_id} ending_reason terminal lacks terminal flag")
    if reason == "scene_boundary" and not scene_boundary:
        errors.append(f"{json_path}: action {action_id} ending_reason scene_boundary lacks scene_boundary flag")
    if reason == "probe_horizon" and horizon != CONTINUATION_CAP:
        errors.append(f"{json_path}: action {action_id} probe_horizon ended after {horizon}, expected {CONTINUATION_CAP}")
    if reason == "probe_horizon" and not truncated:
        errors.append(f"{json_path}: action {action_id} probe_horizon lacks truncation flag")
    if reason == "probe_horizon" and not probe_horizon_reached:
        errors.append(f"{json_path}: action {action_id} probe_horizon lacks probe_horizon_reached")
    if reason == "validity_gap" and not validity_gap:
        errors.append(f"{json_path}: action {action_id} ending_reason validity_gap lacks validity_gap flag")
    if reason == "truncated" and not truncated:
        errors.append(f"{json_path}: action {action_id} ending_reason truncated lacks truncation flag")
    if reason == "censored" and (terminal or scene_boundary or validity_gap or truncated):
        errors.append(f"{json_path}: action {action_id} censored reason conflicts with terminal/boundary flags")


def _verify_eval_metrics(json_path, metadata, arrays, branches, branch_indices, errors):
    schema = metadata.get("metrics_schema") or {}
    schema_keys = schema.get("keys")
    expected_schema = {
        name: f"eval_metrics__{name.replace('/', '__')}" for name in REQUIRED_EVAL_METRICS
    }
    if not isinstance(schema_keys, dict) or schema_keys != expected_schema:
        errors.append(f"{json_path}: metrics_schema does not contain the required 40 metric mappings")
    if len(REQUIRED_EVAL_METRICS) != 40:
        errors.append("verifier bug: required evaluation metric set is not 40 fields")
    for metric, array_name in expected_schema.items():
        if array_name not in arrays or arrays[array_name].shape != (12,):
            errors.append(f"{json_path}: required eval metric array {array_name} is missing or malformed")
            continue
        values = arrays[array_name]
        if values.dtype.kind not in "fiu" or np.any(~np.isfinite(values)):
            errors.append(f"{json_path}: required eval metric array {array_name} is nonnumeric or nonfinite")
        for action_id, branch_index in branch_indices.items():
            metric_record = branches[branch_index].get("metrics_mean")
            if not isinstance(metric_record, dict) or set(metric_record) != REQUIRED_EVAL_METRICS:
                errors.append(f"{json_path}: action {action_id} metrics_mean does not have exactly 40 keys")
                continue
            if metric not in metric_record:
                errors.append(f"{json_path}: action {action_id} lacks required eval metric {metric}")
                continue
            if not _same_scalar(metric_record[metric], values[branch_index]):
                errors.append(f"{json_path}: action {action_id} eval metric {metric} differs from its NPZ array")


def _bin_id(state):
    assigned = state.get("assigned_bin") or {}
    if isinstance(assigned, dict):
        return str(assigned.get("bin_id", "unassigned"))
    return str(assigned)


def _verify_collection_summary(records_dir, selection, probe_metadata, expected_per_bin, run_signature, issues):
    summary_path = records_dir / "collection_summary.json"
    if not summary_path.is_file():
        issues.append(f"collection summary is missing: {summary_path}")
        return
    try:
        summary = _read_json(summary_path)
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"cannot read collection summary {summary_path}: {error}")
        return
    if run_signature and summary.get("run_signature") != run_signature:
        issues.append("collection summary run_signature differs from destination manifest")
    selected_states = selection.get("states", [])
    selected_by_bin = Counter(_bin_id(state) for state in selected_states)
    selected_ids = {str(state.get("state_id", "")) for state in selected_states}
    completed = [
        metadata for state_id, metadata in probe_metadata.items()
        if state_id in selected_ids and metadata.get("status") == "complete"
    ]
    completed_by_bin = Counter()
    completed_branch_count = 0
    eval_metric_keys = set()
    for metadata in completed:
        embedded_state = metadata.get("selection_state") or {}
        assigned = embedded_state.get("assigned_bin", metadata.get("assigned_bin"))
        completed_by_bin[_bin_id({"assigned_bin": assigned})] += 1
        branches = metadata.get("branches") or []
        completed_branch_count += len(branches)
        for branch in branches:
            eval_metric_keys.update((branch.get("metrics_mean") or {}).keys())
    expected_branch_count = len(selected_states) * len(ACTION_IDS)
    if any(count != expected_per_bin for count in selected_by_bin.values()):
        issues.append("collection summary selection inputs do not have the requested per-bin count")
    expected_metric_keys = sorted(REQUIRED_EVAL_METRICS)
    expected_fields = {
        "selection_state_count": len(selected_states),
        "completed_state_count": len(completed),
        "selected_by_bin": dict(sorted(selected_by_bin.items())),
        "completed_by_bin": dict(sorted(completed_by_bin.items())),
        "action_count": len(ACTION_IDS),
        "repeats": 1,
        "continuation_horizon": CONTINUATION_CAP,
        "expected_branch_count": expected_branch_count,
        "completed_branch_count": completed_branch_count,
        "eval_metric_key_count": len(eval_metric_keys),
        "eval_metric_keys": sorted(eval_metric_keys),
        "complete": len(completed) == len(selected_states) and completed_branch_count == expected_branch_count,
    }
    for field, expected in expected_fields.items():
        if summary.get(field) != expected:
            issues.append(f"collection summary {field} differs from recomputed counts")
    if summary.get("eval_metric_key_count") != 40 or sorted(eval_metric_keys) != expected_metric_keys:
        issues.append("collection summary does not report exactly the required 40 evaluation metrics")


def verify_binned_probes(
    destination_model_dir,
    reference_model_dir,
    *,
    expected_per_bin=15,
):
    destination_model_dir = Path(destination_model_dir).expanduser().resolve()
    reference_model_dir = Path(reference_model_dir).expanduser().resolve()
    records_dir = destination_model_dir / "records"
    references_dir = reference_model_dir / "records" / "references"
    report = {
        "destination_model_dir": str(destination_model_dir),
        "reference_model_dir": str(reference_model_dir),
        "expected_per_bin": int(expected_per_bin),
        "continuation_cap": CONTINUATION_CAP,
        "required_eval_metric_count": len(REQUIRED_EVAL_METRICS),
        "selected_state_count": 0,
        "probe_bundle_count": 0,
        "branch_count": 0,
        "bin_counts": {},
        "ending_reason_counts": {},
        "max_observed_horizon": 0,
        "issues": [],
    }
    issues = report["issues"]
    selection_path = records_dir / "selection.json"
    geometry_manifest_path = reference_model_dir / "records" / "geometry_manifest.json"
    destination_manifest_path = records_dir / "manifest.json"
    if not selection_path.is_file():
        issues.append(f"selection file is missing: {selection_path}")
        report["ok"] = False
        return report
    if not references_dir.is_dir():
        issues.append(f"reference records directory is missing: {references_dir}")
        report["ok"] = False
        return report
    try:
        selection = _read_json(selection_path)
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"cannot read selection file {selection_path}: {error}")
        report["ok"] = False
        return report
    excluded_bins = set(selection.get("excluded_assigned_bins", []))
    original_selection = selection
    if excluded_bins:
        selection_path = records_dir / "provenance" / "original_selection.json"
        original_selection = _read_json(selection_path)
        retained = [state for state in original_selection["states"] if _bin_id(state) not in excluded_bins]
        if retained != selection.get("states"):
            issues.append("retained states differ from the original selection minus excluded assigned bins")
    destination_manifest = None
    run_signature = None
    source_env_config = None
    if destination_manifest_path.is_file():
        try:
            destination_manifest = _read_json(destination_manifest_path)
            run_signature = destination_manifest.get("signature")
            source_env_config = (
                destination_manifest.get("resolved_config", {}).get("source_env")
            )
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"cannot read destination manifest: {error}")
    else:
        issues.append(f"destination manifest is missing: {destination_manifest_path}")
    if destination_manifest is not None:
        run_settings = destination_manifest.get("run_settings") or {}
        if run_settings.get("selection_file_sha256") != _sha256_file(selection_path):
            issues.append("destination manifest selection hash differs from selection.json")
        if run_settings.get("source_geometry_manifest_signature") != selection.get(
            "source_geometry_manifest_signature"
        ):
            issues.append("destination manifest source geometry signature differs from selection.json")
    geometry_signature = None
    if geometry_manifest_path.is_file():
        try:
            geometry_signature = _read_json(geometry_manifest_path).get("signature")
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"cannot read source geometry manifest: {error}")
    elif selection.get("source_geometry_manifest_signature"):
        issues.append(f"source geometry manifest is missing: {geometry_manifest_path}")
    selected_signature = selection.get("source_geometry_manifest_signature")
    if geometry_signature and selected_signature != geometry_signature:
        issues.append("selection source_geometry_manifest_signature differs from second-run geometry manifest")
    if selection.get("model") != destination_model_dir.name:
        issues.append("selection model name differs from destination model directory")

    selection_errors, state_by_id, raw_bin_counts, raw_unique_count, source_by_state = _verify_selection_states(
        original_selection, expected_per_bin, references_dir, geometry_signature
    )
    if excluded_bins:
        state_by_id = {key: state for key, state in state_by_id.items() if _bin_id(state) not in excluded_bins}
        raw_unique_count = len({(source_by_state[key]["shard_path"], source_by_state[key]["row_index"])
                                for key in state_by_id})
    issues.extend(selection_errors)
    report["selected_state_count"] = len(state_by_id)
    report["raw_unique_state_count"] = raw_unique_count
    report["bin_counts"] = {
        f"{axis}:{index:02d}": raw_bin_counts.get((axis, index), 0)
        for axis, edges in AXIS_EDGES.items()
        for index in range(len(edges) - 1)
        if f"{axis}:{index:02d}" not in excluded_bins
    }

    probes_dir = records_dir / "probes"
    probe_paths = sorted(probes_dir.glob("*.json")) if probes_dir.is_dir() else []
    if not probes_dir.is_dir():
        issues.append(f"probe records directory is missing: {probes_dir}")
    probe_by_state = {}
    probe_metadata = {}
    ending_reasons = Counter()
    max_horizon = 0
    for json_path in probe_paths:
        try:
            metadata = _read_json(json_path)
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"cannot read probe JSON {json_path}: {error}")
            continue
        state_id = str(metadata.get("state_id", ""))
        if not state_id:
            issues.append(f"{json_path}: missing state_id")
            continue
        if state_id in probe_by_state:
            issues.append(f"duplicate probe bundles for state {state_id}")
        probe_by_state[state_id] = json_path
        probe_metadata[state_id] = metadata
        if run_signature and metadata.get("run_signature") != run_signature:
            issues.append(f"{json_path}: run_signature differs from destination manifest")
        if state_id not in state_by_id:
            issues.append(f"probe bundle {json_path.name} has an unselected state_id {state_id}")
            continue
        bundle_issues, counts = _verify_probe_bundle(
            json_path, state_by_id[state_id], state_id,
            source_info=source_by_state.get(state_id), geometry_signature=geometry_signature,
            source_env_config=source_env_config,
        )
        issues.extend(bundle_issues)
        if counts:
            report["probe_bundle_count"] += 1
            report["branch_count"] += counts["branch_count"]
            max_horizon = max(max_horizon, counts["max_observed_horizon"])
            try:
                metadata = _read_json(json_path)
                for branch in metadata.get("branches", []):
                    ending_reasons[str(branch.get("ending_reason", "missing"))] += 1
            except (OSError, json.JSONDecodeError):
                pass
    missing_probe_ids = sorted(set(state_by_id) - set(probe_by_state))
    if missing_probe_ids:
        issues.append(f"missing probe bundles for {len(missing_probe_ids)} selected states")
    extra_npz = sorted(path.name for path in probes_dir.glob("*.npz")) if probes_dir.is_dir() else []
    known_npz = {path.with_suffix(".npz").name for path in probe_paths}
    if set(extra_npz) != known_npz:
        issues.append("probe directory contains unmatched JSON/NPZ bundle files")
    _verify_collection_summary(
        records_dir, selection, probe_metadata, expected_per_bin, run_signature, issues
    )
    report["ending_reason_counts"] = dict(sorted(ending_reasons.items()))
    report["max_observed_horizon"] = max_horizon
    report["ok"] = not issues
    return report


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination_model_dir", type=Path,
                        help="Model directory under runs/third_20261122, such as .../pql")
    parser.add_argument("reference_model_dir", type=Path,
                        help="Model directory under runs/second_20261122 with reference shards")
    parser.add_argument("--expected-per-bin", type=int, default=15)
    parser.add_argument("--report", type=Path,
                        help="Optional JSON report destination; otherwise print to stdout")
    return parser


def main(argv=None):
    arguments = build_argument_parser().parse_args(argv)
    if arguments.expected_per_bin <= 0:
        raise SystemExit("--expected-per-bin must be positive")
    report = verify_binned_probes(
        arguments.destination_model_dir,
        arguments.reference_model_dir,
        expected_per_bin=arguments.expected_per_bin,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
