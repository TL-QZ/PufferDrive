"""Choose deterministic, diverse states for the third Q-calibration run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .run_layout import PROJECT_DIR, initialize_model_directory, model_directory, records_directory


DEFAULT_SEED = 20261122
REQUESTED_PER_BIN = 15
FORMAT_VERSION = 1
SELECTION_ALGORITHM = "balanced-map-episode-agent-no-adjacent-frame-v1"

GEOMETRY_FIELDS = (
    "nearest_observed_partner_center_distance_m",
    "nearest_observed_road_boundary_distance_m",
    "nearest_visible_goal_distance_m",
    "current_goal_distance_m",
)

# Values at an edge belong to the bin starting at that edge. The final bin is
# open-ended; JSON uses null for its unbounded upper limit.
PRIMARY_AXES = {
    "nearest_observed_partner_center_distance_m": (0.0, 5.0, 10.0, 20.0, 40.0, 80.0, np.inf),
    "nearest_observed_road_boundary_distance_m": (0.0, 1.5, 2.0, 3.0, 5.0, 8.0, np.inf),
    "nearest_visible_goal_distance_m": (0.0, 2.0, 5.0, 12.0, 30.0, 100.0, 300.0, np.inf),
}

REQUIRED_ROW_FIELDS = (
    "frame_idx",
    "agent_id",
    "slot",
    "current_goal_idx",
    "goal_count",
    "eligible",
    *GEOMETRY_FIELDS,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        required=True,
        help="Second-run model directory containing records/references and geometry_manifest.json",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Output model directory under runs/<run>/<model>",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Deterministic selection seed (default: {DEFAULT_SEED})")
    parser.add_argument("--per-bin", type=int, default=REQUESTED_PER_BIN,
                        help=f"Unique states assigned to each bin (default: {REQUESTED_PER_BIN})")
    return parser


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _axis_bin_index(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Return lower-inclusive bin IDs, or -1 for missing/non-finite values."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Distance values must be one-dimensional")
    if np.any(np.isfinite(values) & (values < 0.0)):
        raise ValueError("Distance values must be non-negative")
    bin_indices = np.searchsorted(np.asarray(edges[1:-1]), values, side="right").astype(np.int16)
    bin_indices[~np.isfinite(values)] = -1
    return bin_indices


def _interval(axis: str, bin_index: int) -> dict[str, Any]:
    edges = PRIMARY_AXES[axis]
    lower_m = float(edges[bin_index])
    upper_m = None if np.isinf(edges[bin_index + 1]) else float(edges[bin_index + 1])
    return {
        "bin_index": int(bin_index),
        "bin_id": f"{axis}:{bin_index:02d}",
        "lower_m": lower_m,
        "upper_m": upper_m,
    }


def _axes_manifest() -> dict[str, dict[str, Any]]:
    result = {}
    for axis, edges in PRIMARY_AXES.items():
        result[axis] = {
            "distance_field": axis,
            "edges_m": [_json_number(value) for value in edges],
            "interval_convention": "lower-inclusive, upper-exclusive; final bin is unbounded",
            "bins": [_interval(axis, index) for index in range(len(edges) - 1)],
        }
    return result


def _required_array(archive, field: str, *, row_count: int | None = None) -> np.ndarray:
    if field not in archive.files:
        raise ValueError(f"Geometry shard is missing required array {field}")
    values = archive[field]
    if row_count is not None and values.shape != (row_count,):
        raise ValueError(f"Geometry shard row field {field} has invalid shape {values.shape}")
    return values


def _load_candidate_pool(source_records_dir: Path) -> dict[str, Any]:
    """Load eligible geometry rows and validate their sidecars and shard hashes."""
    manifest_path = source_records_dir / "geometry_manifest.json"
    references_dir = source_records_dir / "references"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Second-run geometry manifest not found: {manifest_path}")
    if not references_dir.is_dir():
        raise FileNotFoundError(f"Second-run geometry references not found: {references_dir}")

    geometry_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_signature = geometry_manifest.get("signature")
    if not isinstance(run_signature, str) or not run_signature:
        raise ValueError(f"Geometry manifest lacks a signature: {manifest_path}")

    sidecar_paths = sorted(references_dir.glob("*/episode_*.json"))
    if not sidecar_paths:
        raise ValueError(f"No per-episode geometry sidecars found under {references_dir}")

    shard_rows: list[dict[str, Any]] = []
    row_pieces: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "map_index", "episode_index", "frame_idx", "agent_id", "slot",
            "current_goal_idx", "goal_count", "source_shard_index", *GEOMETRY_FIELDS,
        )
    }
    bin_pieces: dict[str, list[np.ndarray]] = {axis: [] for axis in PRIMARY_AXES}
    map_identity: dict[str, tuple[int, str]] = {}
    shard_manifest_rows = []

    for sidecar_path in sidecar_paths:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        map_name = metadata.get("map_name")
        if not isinstance(map_name, str) or sidecar_path.parent.name != map_name:
            raise ValueError(f"Episode sidecar map identity does not match its directory: {sidecar_path}")
        if metadata.get("run_signature") != run_signature:
            raise ValueError(f"Episode sidecar belongs to another geometry run: {sidecar_path}")
        episode_index = metadata.get("episode_index")
        map_index = metadata.get("map_index")
        if not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError(f"Episode sidecar has invalid episode_index: {sidecar_path}")
        if not isinstance(map_index, int) or map_index < 0:
            raise ValueError(f"Episode sidecar has invalid map_index: {sidecar_path}")
        map_path = metadata.get("map_path")
        if not isinstance(map_path, str) or not map_path:
            raise ValueError(f"Episode sidecar has invalid map_path: {sidecar_path}")
        prior_map = map_identity.setdefault(map_name, (map_index, map_path))
        if prior_map != (map_index, map_path):
            raise ValueError(f"Map metadata changes across geometry episodes for {map_name}")

        data_file = metadata.get("data_file")
        expected_data_name = sidecar_path.with_suffix(".npz").name
        if data_file != expected_data_name:
            raise ValueError(f"Episode sidecar has unexpected data_file {data_file!r}: {sidecar_path}")
        data_path = sidecar_path.parent / data_file
        if not data_path.is_file():
            raise FileNotFoundError(f"Geometry shard not found: {data_path}")
        actual_data_hash = _sha256_file(data_path)
        if metadata.get("data_sha256") != actual_data_hash:
            raise ValueError(f"Geometry shard hash differs from its sidecar: {data_path}")

        with np.load(data_path, allow_pickle=False) as archive:
            row_count = int(archive["frame_idx"].size) if "frame_idx" in archive.files else -1
            if row_count < 0:
                raise ValueError(f"Geometry shard lacks frame_idx: {data_path}")
            arrays = {field: _required_array(archive, field, row_count=row_count)
                      for field in REQUIRED_ROW_FIELDS}
            frame_count = int(metadata.get("frame_count", -1))
            if frame_count <= 0:
                raise ValueError(f"Episode sidecar has invalid frame_count: {sidecar_path}")
            fingerprints = _required_array(archive, "fingerprints")
            observation_fingerprints = _required_array(archive, "observation_fingerprints")
            if fingerprints.shape != (frame_count,) or observation_fingerprints.shape != (frame_count,):
                raise ValueError(f"Geometry shard fingerprint arrays have invalid shape: {data_path}")
            if arrays["eligible"].dtype != np.bool_:
                raise ValueError(f"Geometry shard eligible array must be bool: {data_path}")
            if not np.issubdtype(arrays["frame_idx"].dtype, np.integer):
                raise ValueError(f"Geometry shard frame_idx must be integer: {data_path}")
            if not np.issubdtype(arrays["agent_id"].dtype, np.integer) or not np.issubdtype(
                arrays["slot"].dtype, np.integer
            ):
                raise ValueError(f"Geometry shard agent_id and slot must be integers: {data_path}")
            eligible = arrays["eligible"]
            if np.any((arrays["frame_idx"] < 0) | (arrays["frame_idx"] >= frame_count)):
                raise ValueError(f"Geometry shard frame_idx is out of range: {data_path}")
            if np.any(eligible & ((arrays["agent_id"] < 0) | (arrays["slot"] < 0))):
                raise ValueError(f"Eligible geometry row has invalid agent_id or slot: {data_path}")
            for field in GEOMETRY_FIELDS:
                values = np.asarray(arrays[field], dtype=np.float64)
                if np.any(np.isinf(values[eligible])) or np.any(np.isfinite(values[eligible]) & (values[eligible] < 0)):
                    raise ValueError(f"Geometry field {field} has an invalid eligible distance: {data_path}")

            source_shard = data_path.relative_to(source_records_dir).as_posix()
            source_metadata = sidecar_path.relative_to(source_records_dir).as_posix()
            shard = {
                "map_index": map_index,
                "map_name": map_name,
                "map_path": map_path,
                "episode_index": episode_index,
                "layout_seed": int(metadata["layout_seed"]),
                "layout_draw_index": int(metadata["layout_draw_index"]),
                "scene_seed": int(metadata["scene_seed"]),
                "episode_seed": int(metadata["episode_seed"]),
                "agent_count": int(metadata["agent_count"]),
                "frame_count": frame_count,
                "source_shard": source_shard,
                "source_metadata": source_metadata,
                "data_sha256": actual_data_hash,
                "fingerprints": np.asarray(fingerprints),
                "observation_fingerprints": np.asarray(observation_fingerprints),
            }
            shard_index = len(shard_rows)
            shard_rows.append(shard)
            shard_manifest_rows.append({
                "source_shard": source_shard,
                "source_metadata": source_metadata,
                "data_sha256": actual_data_hash,
            })

            valid_rows = np.flatnonzero(eligible)
            if valid_rows.size == 0:
                continue
            constant_values = {
                "map_index": map_index,
                "episode_index": episode_index,
                "source_shard_index": shard_index,
            }
            for field, value in constant_values.items():
                row_pieces[field].append(np.full(valid_rows.size, value, dtype=np.int32))
            for field in (
                "frame_idx", "agent_id", "slot", "current_goal_idx", "goal_count",
            ):
                row_pieces[field].append(np.asarray(arrays[field][valid_rows], dtype=np.int32))
            for field in GEOMETRY_FIELDS:
                values = np.asarray(arrays[field][valid_rows], dtype=np.float32)
                row_pieces[field].append(values)
            for axis in PRIMARY_AXES:
                bin_pieces[axis].append(_axis_bin_index(arrays[axis][valid_rows], PRIMARY_AXES[axis]))

    if not row_pieces["frame_idx"]:
        raise ValueError(f"No eligible geometry rows found under {references_dir}")
    rows = {name: np.concatenate(pieces) for name, pieces in row_pieces.items()}
    bin_indices = {axis: np.concatenate(pieces) for axis, pieces in bin_pieces.items()}
    if np.any(rows["goal_count"] < 0):
        raise ValueError("Eligible geometry rows contain a negative goal_count")

    identity = np.rec.fromarrays(
        [rows[name] for name in ("map_index", "episode_index", "frame_idx", "agent_id", "slot")],
        names="map,episode,frame,agent,slot",
    )
    _, identity_counts = np.unique(identity, return_counts=True)
    if np.any(identity_counts > 1):
        raise ValueError("Source geometry records contain duplicate map/episode/frame/agent/slot states")

    return {
        "rows": rows,
        "bin_indices": bin_indices,
        "shards": shard_rows,
        "source_shards": shard_manifest_rows,
        "source_geometry_manifest_signature": run_signature,
    }


def _factor_group_ids(*arrays: np.ndarray) -> np.ndarray:
    identity = np.rec.fromarrays(arrays, names=[f"field_{index}" for index in range(len(arrays))])
    _, inverse = np.unique(identity, return_inverse=True)
    return inverse.astype(np.int32, copy=False)


def _choose_least_loaded_group(
    candidates: np.ndarray,
    group_ids: np.ndarray,
    within_bin_load: dict[int, int],
    global_load: dict[int, int],
    rng: np.random.Generator,
) -> int:
    available_groups = np.unique(group_ids[candidates])
    if not available_groups.size:
        raise RuntimeError("Cannot choose a group from an empty candidate set")
    return min(
        (int(group) for group in available_groups),
        key=lambda group: (
            within_bin_load.get(group, 0),
            global_load.get(group, 0),
            float(rng.random()),
            group,
        ),
    )


def _select_assignments(pool: dict[str, Any], *, seed: int, per_bin: int) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    rows = pool["rows"]
    row_count = rows["frame_idx"].size
    rng = np.random.default_rng(seed)
    map_groups = rows["map_index"].astype(np.int32, copy=False)
    episode_groups = _factor_group_ids(rows["map_index"], rows["episode_index"])
    car_groups = _factor_group_ids(rows["map_index"], rows["episode_index"], rows["agent_id"], rows["slot"])
    frame_base = int(rows["frame_idx"].max()) + 2
    car_frame_keys = car_groups.astype(np.int64) * frame_base + rows["frame_idx"].astype(np.int64)

    selected_mask = np.zeros(row_count, dtype=np.bool_)
    blocked_car_frame_keys: set[int] = set()
    global_map_load: dict[int, int] = {}
    global_episode_load: dict[int, int] = {}
    global_car_load: dict[int, int] = {}
    assignments: list[dict[str, Any]] = []
    bin_summary: dict[str, list[dict[str, Any]]] = {}
    adjacency_relaxed_count = 0

    for axis, edges in PRIMARY_AXES.items():
        axis_summary = []
        for bin_index in range(len(edges) - 1):
            candidates = np.flatnonzero(pool["bin_indices"][axis] == bin_index)
            candidate_count = int(candidates.size)
            within_map_load: dict[int, int] = {}
            within_episode_load: dict[int, int] = {}
            within_car_load: dict[int, int] = {}
            selected_count = 0
            for _ in range(per_bin):
                available = candidates[~selected_mask[candidates]]
                if not available.size:
                    break
                blocked = np.fromiter(blocked_car_frame_keys, dtype=np.int64, count=len(blocked_car_frame_keys))
                non_adjacent = available[~np.isin(car_frame_keys[available], blocked)] if blocked.size else available
                if non_adjacent.size:
                    choices = non_adjacent
                else:
                    # Quota takes priority when the remaining support consists only
                    # of a car at a neighboring frame to an earlier selected state.
                    choices = available
                    adjacency_relaxed_count += 1

                chosen_map = _choose_least_loaded_group(
                    choices, map_groups, within_map_load, global_map_load, rng,
                )
                in_map = choices[map_groups[choices] == chosen_map]
                chosen_episode = _choose_least_loaded_group(
                    in_map, episode_groups, within_episode_load, global_episode_load, rng,
                )
                in_episode = in_map[episode_groups[in_map] == chosen_episode]
                chosen_car = _choose_least_loaded_group(
                    in_episode, car_groups, within_car_load, global_car_load, rng,
                )
                in_car = in_episode[car_groups[in_episode] == chosen_car]
                selected_row = int(rng.choice(in_car))
                selected_mask[selected_row] = True
                selected_count += 1
                within_map_load[chosen_map] = within_map_load.get(chosen_map, 0) + 1
                within_episode_load[chosen_episode] = within_episode_load.get(chosen_episode, 0) + 1
                within_car_load[chosen_car] = within_car_load.get(chosen_car, 0) + 1
                global_map_load[chosen_map] = global_map_load.get(chosen_map, 0) + 1
                global_episode_load[chosen_episode] = global_episode_load.get(chosen_episode, 0) + 1
                global_car_load[chosen_car] = global_car_load.get(chosen_car, 0) + 1
                selected_frame = int(rows["frame_idx"][selected_row])
                for nearby_frame in (selected_frame - 1, selected_frame, selected_frame + 1):
                    if nearby_frame >= 0:
                        blocked_car_frame_keys.add(int(chosen_car) * frame_base + nearby_frame)
                assignments.append({"row_index": selected_row, "axis": axis, "bin_index": bin_index})

            tag = _interval(axis, bin_index)
            axis_summary.append({
                **tag,
                "candidate_count": candidate_count,
                "selected_count": selected_count,
                "shortage_count": max(0, per_bin - selected_count),
            })
        bin_summary[axis] = axis_summary

    return assignments, bin_summary, adjacency_relaxed_count


def _tag_for_axis(axis: str, bin_index: int) -> dict[str, Any]:
    if bin_index < 0:
        return {
            "bin_index": None,
            "bin_id": None,
            "lower_m": None,
            "upper_m": None,
            "missing": True,
        }
    return _interval(axis, bin_index)


def _selected_state(pool: dict[str, Any], assignment: dict[str, Any]) -> dict[str, Any]:
    row_index = assignment["row_index"]
    rows = pool["rows"]
    shard = pool["shards"][int(rows["source_shard_index"][row_index])]
    bin_tags = {
        axis: _tag_for_axis(axis, int(pool["bin_indices"][axis][row_index]))
        for axis in PRIMARY_AXES
    }
    assigned_tag = _interval(assignment["axis"], assignment["bin_index"])
    distances = {
        field: _json_number(rows[field][row_index])
        for field in GEOMETRY_FIELDS
    }
    frame_idx = int(rows["frame_idx"][row_index])
    agent_id = int(rows["agent_id"][row_index])
    slot = int(rows["slot"][row_index])
    episode_index = int(rows["episode_index"][row_index])
    map_name = shard["map_name"]
    scene_key = (
        f"{map_name}/episode_{episode_index:05d}/frame_{frame_idx:06d}/"
        f"agent_{agent_id:04d}_slot_{slot:04d}"
    )
    return {
        "state_id": scene_key,
        "scene_key": scene_key,
        "assigned_bin": {"axis": assignment["axis"], **assigned_tag},
        "bin_tags": bin_tags,
        "map_index": int(rows["map_index"][row_index]),
        "map_name": map_name,
        "map_path": shard["map_path"],
        "episode_index": episode_index,
        "frame_idx": frame_idx,
        "agent_id": agent_id,
        "slot": slot,
        "current_goal_idx": int(rows["current_goal_idx"][row_index]),
        "goal_count": int(rows["goal_count"][row_index]),
        "layout_seed": shard["layout_seed"],
        "layout_draw_index": shard["layout_draw_index"],
        "scene_seed": shard["scene_seed"],
        "episode_seed": shard["episode_seed"],
        "agent_count": shard["agent_count"],
        "source_shard": shard["source_shard"],
        "source_metadata": shard["source_metadata"],
        "fingerprint": str(shard["fingerprints"][frame_idx]),
        "observation_fingerprint": str(shard["observation_fingerprints"][frame_idx]),
        "distances_m": distances,
    }


def build_selection(
    pool: dict[str, Any],
    *,
    model: str,
    source_records_dir: str,
    seed: int = DEFAULT_SEED,
    per_bin: int = REQUESTED_PER_BIN,
) -> dict[str, Any]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if per_bin <= 0:
        raise ValueError("per_bin must be positive")
    assignments, bin_summary, adjacency_relaxed_count = _select_assignments(
        pool, seed=seed, per_bin=per_bin,
    )
    states = [_selected_state(pool, assignment) for assignment in assignments]
    shortages = []
    for axis, bins in bin_summary.items():
        for summary in bins:
            if summary["shortage_count"]:
                shortages.append({
                    "axis": axis,
                    "bin_index": summary["bin_index"],
                    "bin_id": summary["bin_id"],
                    "lower_m": summary["lower_m"],
                    "upper_m": summary["upper_m"],
                    "requested_count": per_bin,
                    "selected_count": summary["selected_count"],
                    "shortage_count": summary["shortage_count"],
                })
    signature_inputs = {
        "model": model,
        "seed": seed,
        "per_bin": per_bin,
        "algorithm": SELECTION_ALGORITHM,
        "source_geometry_manifest_signature": pool["source_geometry_manifest_signature"],
        "source_shards": pool["source_shards"],
        "axes": _axes_manifest(),
    }
    signature = _canonical_sha256(signature_inputs)
    return {
        "format_version": FORMAT_VERSION,
        "signature": signature,
        "model": model,
        "seed": seed,
        "algorithm": SELECTION_ALGORITHM,
        "source_records_dir": source_records_dir,
        "source_geometry_manifest_signature": pool["source_geometry_manifest_signature"],
        "source_shards": pool["source_shards"],
        "axes": _axes_manifest(),
        "requested_per_bin": per_bin,
        "requested_state_count": per_bin * sum(len(edges) - 1 for edges in PRIMARY_AXES.values()),
        "selected_state_count": len(states),
        "bin_summary": bin_summary,
        "shortages": shortages,
        "adjacency_relaxed_count": adjacency_relaxed_count,
        "states": states,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as stream:
            temp_name = stream.name
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_selection(path: Path, selection: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("signature") != selection["signature"]:
            raise ValueError(
                f"Existing selection at {path} was built from different inputs; choose a new model-dir"
            )
        selection["created_at"] = existing.get("created_at", "")
        if not selection["created_at"]:
            selection["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        selection["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _atomic_write_json(path, selection)


def run_selection(args) -> dict[str, Any]:
    source_model_dir = model_directory(args.source_model_dir)
    output_model_dir = model_directory(args.model_dir)
    if source_model_dir == output_model_dir:
        raise ValueError("--model-dir must be separate from --source-model-dir")
    if source_model_dir.name != output_model_dir.name:
        raise ValueError("source and output model directories must have the same model name")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.per_bin <= 0:
        raise ValueError("--per-bin must be positive")

    source_records_dir = records_directory(source_model_dir)
    try:
        source_records_relative = source_records_dir.relative_to(PROJECT_DIR).as_posix()
    except ValueError as error:
        raise ValueError("--source-model-dir must be inside this Q-calibration project") from error
    pool = _load_candidate_pool(source_records_dir)
    selection = build_selection(
        pool,
        model=source_model_dir.name,
        source_records_dir=source_records_relative,
        seed=args.seed,
        per_bin=args.per_bin,
    )
    output_model_dir = initialize_model_directory(output_model_dir)
    selection_path = records_directory(output_model_dir) / "selection.json"
    _write_selection(selection_path, selection)
    print(json.dumps({
        "model": selection["model"],
        "selection": str(selection_path),
        "selected_state_count": selection["selected_state_count"],
        "requested_state_count": selection["requested_state_count"],
        "shortages": selection["shortages"],
        "adjacency_relaxed_count": selection["adjacency_relaxed_count"],
        "signature": selection["signature"],
    }, sort_keys=True))
    return selection


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    selection = run_selection(args)
    if selection["shortages"]:
        raise RuntimeError(f"Selection did not fill every requested bin: {selection['shortages']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
