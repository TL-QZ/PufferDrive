"""Replay the saved reference scenes and resolve their first unfulfilled route goal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from . import select_binned_probes as selector
from .run_layout import initialize_model_directory, model_directory, records_directory
from .run_probes import observation_fingerprint


RUN_ROOT = Path("project/soft_actor_critic_q_calibration/runs")
DEFAULT_RUN = RUN_ROOT / "fourth_20261124"
SOURCE_RUN = RUN_ROOT / "second_20261122"
DEFAULT_PER_BIN = 20
CORRECTION_ALGORITHM = "drive-swept-segment-ordered-goals-v1"
GOAL_HEIGHT_TOLERANCE_M = 4.0
GOAL_RADIUS_COEFFICIENT_INDEX = 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--per-bin", type=int, default=DEFAULT_PER_BIN)
    parser.add_argument("--seed", type=int, default=selector.DEFAULT_SEED)
    return parser


def _source_config(source_records: Path):
    manifest = json.loads((source_records / "geometry_manifest.json").read_text(encoding="utf-8"))
    settings = manifest["settings"]
    env_config = dict(settings["resolved_env_config"])
    map_dir = Path(env_config["map_dir"]).expanduser()
    if not map_dir.is_absolute():
        map_dir = Path(__file__).resolve().parents[3] / map_dir
    env_config["map_dir"] = str(map_dir.resolve())
    return manifest, settings, env_config


def _sidecars(source_records: Path, map_index: int | None):
    paths = sorted((source_records / "references").glob("*/episode_*.json"))
    if map_index is None:
        return paths
    return [path for path in paths if int(json.loads(path.read_text())["map_index"]) == map_index]


def _read_goal_geometry(batch):
    result = batch.binding.vec_get_goal_geometry(batch.c_envs)
    return {name: np.asarray(values) for name, values in result.items()}


def _goal_distance_comparison(source_records: Path, corrections_records: Path):
    edges = selector.PRIMARY_AXES["nearest_visible_goal_distance_m"]
    old_bins = np.zeros(len(edges) - 1, dtype=np.int64)
    corrected_bins = np.zeros(len(edges) - 1, dtype=np.int64)
    source_missing = 0
    corrected_missing = 0
    changed_distances = 0
    changed_bins = 0
    row_count = 0
    for sidecar_path in sorted(corrections_records.glob("*/episode_*.json")):
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        corrected_path = sidecar_path.with_suffix(".npz")
        source_path = source_records / metadata["source_shard"]
        with np.load(source_path, allow_pickle=False) as source_archive:
            old_values = np.asarray(source_archive["nearest_visible_goal_distance_m"], dtype=np.float64)
        with np.load(corrected_path, allow_pickle=False) as corrected_archive:
            new_values = np.asarray(corrected_archive["next_goal_distance_m"], dtype=np.float64)
        if old_values.shape != new_values.shape:
            raise ValueError(f"Source and corrected goal arrays differ in shape: {corrected_path}")
        old_finite = np.isfinite(old_values)
        new_finite = np.isfinite(new_values)
        same_finite = old_finite == new_finite
        distance_changed = ~same_finite
        both_finite = old_finite & new_finite
        distance_changed[both_finite] = np.abs(old_values[both_finite] - new_values[both_finite]) > 1e-4
        changed_distances += int(np.count_nonzero(distance_changed))
        old_index = selector._axis_bin_index(old_values.astype(np.float32), edges)
        new_index = selector._axis_bin_index(new_values.astype(np.float32), edges)
        changed_bins += int(np.count_nonzero(old_index != new_index))
        if np.any(old_finite):
            old_bins += np.bincount(old_index[old_finite], minlength=len(old_bins))
        if np.any(new_finite):
            corrected_bins += np.bincount(new_index[new_finite], minlength=len(corrected_bins))
        source_missing += int(np.count_nonzero(~old_finite))
        corrected_missing += int(np.count_nonzero(~new_finite))
        row_count += int(old_values.size)
    bin_labels = [selector._interval("nearest_visible_goal_distance_m", idx)["bin_id"]
                  for idx in range(len(old_bins))]
    return {
        "eligible_agent_frame_count": row_count,
        "agent_frames_with_changed_goal_distance": changed_distances,
        "agent_frames_with_changed_goal_bin": changed_bins,
        "source_frames_without_visible_goal": source_missing,
        "corrected_frames_without_unfulfilled_goal": corrected_missing,
        "source_goal_bin_counts": dict(zip(bin_labels, old_bins.astype(int).tolist())),
        "corrected_goal_bin_counts": dict(zip(bin_labels, corrected_bins.astype(int).tolist())),
    }


def _collect_episode(source_records: Path, metadata: dict, output_records: Path, env_config: dict):
    from . import probe_env

    source_path = source_records / "references" / metadata["map_name"] / metadata["data_file"]
    if _sha256_file(source_path) != metadata["data_sha256"]:
        raise ValueError(f"Reference shard hash mismatch: {source_path}")
    with np.load(source_path, allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}

    frame_count = int(metadata["frame_count"])
    agent_count = int(metadata["agent_count"])
    if source["joint_actions"].shape != (frame_count, agent_count):
        raise ValueError(f"Reference action array has wrong shape: {source_path}")
    if source["preaction_eligible"].shape != (frame_count, agent_count):
        raise ValueError(f"Reference eligibility array has wrong shape: {source_path}")

    template = probe_env.create_template(env_config, seed=int(metadata["layout_seed"]))
    try:
        map_path = str(Path(metadata["map_path"]).resolve())
        if int(template.map_agent_counts.get(map_path, -1)) != agent_count:
            raise ValueError(f"Replayed training layout has different agent count: {source_path}")
        batch = probe_env.SceneBatch.from_template(
            template, map_path, agent_count, copies=1, scene_seed=int(metadata["scene_seed"])
        )
    finally:
        template.close()

    corrected = {name: [] for name in (
        "agent_id", "frame_idx", "slot", "current_goal_idx", "resolved_goal_idx",
        "goal_count", "fulfilled_goal_count", "goal_radius_m", "next_goal_distance_m",
        "previous_pose_xy", "pose_xyz", "current_goal_xyz", "resolved_goal_xyz",
    )}
    try:
        batch.reset()
        for frame_idx in range(frame_count):
            frame_fingerprint = batch.fingerprint(0)
            if frame_fingerprint != str(source["fingerprints"][frame_idx]):
                raise ValueError(f"Native state replay differs at {metadata['map_name']} frame {frame_idx}")
            frame_observation = np.asarray(batch.observations[0], dtype=np.float32)
            if observation_fingerprint(frame_observation) != str(source["observation_fingerprints"][frame_idx]):
                raise ValueError(f"Observation replay differs at {metadata['map_name']} frame {frame_idx}")

            route = _read_goal_geometry(batch)
            row_count = agent_count
            if route["agent_id"].shape != (row_count,):
                raise ValueError("Native goal geometry does not match the replayed agent slots")
            route_current = route["current_goal_idx"].astype(np.int32)
            route_count = route["goal_count"].astype(np.int32)
            resolved = route["resolved_goal_idx"].astype(np.int32)
            fulfilled = route["fulfilled_goal_count"].astype(np.int32)
            radius = route["goal_radius_m"].astype(np.float32)
            next_distance = route["next_goal_distance_m"].astype(np.float32)
            previous_xy = route["prev_xy"].astype(np.float32).reshape(agent_count, 2)
            pose_xyz = route["pose_xyz"].astype(np.float32).reshape(agent_count, 3)
            route_goals = route["route_goals_xyz"].astype(np.float32).reshape(agent_count, -1, 3)
            preaction_eligible = np.asarray(source["preaction_eligible"][frame_idx], dtype=np.bool_)
            slots = np.flatnonzero(preaction_eligible)
            source_rows = np.flatnonzero(source["frame_idx"] == frame_idx)
            if not np.array_equal(route["agent_id"][slots], source["agent_id"][source_rows]):
                raise ValueError(f"Native agent identities differ from the reference at frame {frame_idx}")
            if not np.array_equal(route_current[slots], source["current_goal_idx"][source_rows]):
                raise ValueError(f"Native current goal indices differ from the reference at frame {frame_idx}")
            if not np.array_equal(route_count[slots], source["goal_count"][source_rows]):
                raise ValueError(f"Native goal counts differ from the reference at frame {frame_idx}")
            for name, values in (
                ("agent_id", route["agent_id"].astype(np.int32)),
                ("current_goal_idx", route_current),
                ("resolved_goal_idx", resolved),
                ("goal_count", route_count),
                ("fulfilled_goal_count", fulfilled),
                ("goal_radius_m", radius),
                ("next_goal_distance_m", next_distance),
            ):
                corrected[name].append(values[slots])
            corrected["frame_idx"].append(np.full(slots.size, frame_idx, dtype=np.int32))
            corrected["slot"].append(slots.astype(np.int32))
            corrected["previous_pose_xy"].append(previous_xy[slots])
            corrected["pose_xyz"].append(pose_xyz[slots])
            current_goals = np.full((slots.size, 3), np.nan, dtype=np.float32)
            resolved_goals = np.full((slots.size, 3), np.nan, dtype=np.float32)
            for local_row, slot in enumerate(slots):
                current_idx = int(route_current[slot])
                resolved_idx = int(resolved[slot])
                if current_idx < route_goals.shape[1] and current_idx < route_count[slot]:
                    current_goals[local_row] = route_goals[slot, current_idx]
                if resolved_idx < route_goals.shape[1] and resolved_idx < route_count[slot]:
                    resolved_goals[local_row] = route_goals[slot, resolved_idx]
            corrected["current_goal_xyz"].append(current_goals)
            corrected["resolved_goal_xyz"].append(resolved_goals)

            step_result = batch.step(batch.action_table[source["joint_actions"][frame_idx]][None, ...])
            if frame_idx + 1 < frame_count and bool(step_result["metadata"]["scene_done"][0]):
                raise ValueError(f"Source episode ended before its saved frame count: {source_path}")
    finally:
        batch.close()

    corrected_arrays = {name: np.concatenate(values, axis=0) for name, values in corrected.items()}
    for field in ("frame_idx", "slot", "agent_id"):
        expected = source[field].astype(np.int32)
        if not np.array_equal(corrected_arrays[field], expected):
            raise ValueError(f"Corrected goal rows do not match source geometry field {field}: {source_path}")
    output_dir = output_records / "goal_distance_corrections" / metadata["map_name"]
    output_path = output_dir / Path(metadata["data_file"]).name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **corrected_arrays)
    sidecar = {
        "format_version": 1,
        "algorithm": CORRECTION_ALGORITHM,
        "source_geometry_signature": metadata["run_signature"],
        "source_shard": source_path.relative_to(source_records).as_posix(),
        "source_shard_sha256": metadata["data_sha256"],
        "map_name": metadata["map_name"],
        "map_index": int(metadata["map_index"]),
        "episode_index": int(metadata["episode_index"]),
        "frame_count": frame_count,
        "eligible_row_count": int(corrected_arrays["frame_idx"].size),
        "goal_hit_rule": {
            "horizontal_distance": "minimum distance from waypoint XY to the native prev_xy -> sim_xy swept segment",
            "strict_radius_comparison": "distance < per-agent reward coefficient REWARD_COEF_GOAL_RADIUS",
            "strict_height_comparison": "abs(sim_z - waypoint_z) < Z_BUFFER (4.0 m)",
            "advance_order": "repeat over consecutive route waypoints until first waypoint not hit by the starting swept segment",
            "binned_distance": "horizontal Euclidean distance from current sim_xy to the resolved waypoint",
        },
        "data_file": output_path.name,
        "data_sha256": _sha256_file(output_path),
    }
    _write_json(output_path.with_suffix(".json"), sidecar)
    return output_path, corrected_arrays


def collect_corrected_references(source_model_dir: Path, output_model_dir: Path):
    source_model_dir = model_directory(source_model_dir)
    output_model_dir = model_directory(output_model_dir)
    if source_model_dir.name not in {"pql", "tes", "ppo"} or output_model_dir.name != source_model_dir.name:
        raise ValueError("source and output model directories must end in the same pql, tes, or ppo name")
    source_records = records_directory(source_model_dir)
    output_records = records_directory(initialize_model_directory(output_model_dir))
    source_manifest, settings, env_config = _source_config(source_records)
    source_manifest_path = source_records / "geometry_manifest.json"
    sidecars = _sidecars(source_records, map_index=None)
    if not sidecars:
        raise FileNotFoundError(f"No source reference sidecars under {source_records}")
    expected_frames_by_map = {}
    for sidecar_path in sidecars:
        source_metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        map_name = source_metadata["map_name"]
        expected_frames_by_map[map_name] = expected_frames_by_map.get(map_name, 0) + int(source_metadata["frame_count"])

    accessor = Path(__file__).parent / "native_probe" / "goal_geometry.h"
    run_settings = {
        "source_model_dir": str(source_model_dir),
        "source_geometry_signature": source_manifest["signature"],
        "source_geometry_manifest_sha256": _sha256_file(source_manifest_path),
        "correction_algorithm": CORRECTION_ALGORITHM,
        "goal_height_tolerance_m": GOAL_HEIGHT_TOLERANCE_M,
        "goal_radius_coefficient_index": GOAL_RADIUS_COEFFICIENT_INDEX,
        "native_accessor_sha256": _sha256_file(accessor),
        "frame_count_by_map": expected_frames_by_map,
    }
    manifest_path = output_records / "goal_distance_corrections" / "manifest.json"
    signature = _canonical_sha256(run_settings)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError(f"Existing correction manifest has different inputs: {manifest_path}")
    else:
        _write_json(manifest_path, {"format_version": 1, "signature": signature, "settings": run_settings})

    progress_path = output_records / "goal_distance_corrections" / "progress.json"
    from . import probe_env
    probe_env.ensure_private_binding()
    completed = []
    for sidecar_path in sidecars:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        source_map = source_records / "references" / metadata["map_name"]
        source_npz = source_map / metadata["data_file"]
        output_npz = output_records / "goal_distance_corrections" / metadata["map_name"] / metadata["data_file"]
        if output_npz.is_file() and output_npz.with_suffix(".json").is_file():
            cached = json.loads(output_npz.with_suffix(".json").read_text(encoding="utf-8"))
            if cached.get("source_shard_sha256") == metadata["data_sha256"] and cached.get("data_sha256") == _sha256_file(output_npz):
                completed.append(metadata["data_file"])
                continue
            raise ValueError(f"Existing corrected geometry shard does not match source: {output_npz}")
        print(f"{output_model_dir.name} {metadata['map_name']} episode {metadata['episode_index']}: replaying {metadata['frame_count']} frames", flush=True)
        _, corrected = _collect_episode(source_records, metadata, output_records, env_config)
        fulfilled = corrected["fulfilled_goal_count"]
        print(json.dumps({
            "map": metadata["map_name"], "episode": metadata["episode_index"],
            "eligible_rows": int(fulfilled.size),
            "rows_skipping_goal": int(np.count_nonzero(fulfilled)),
            "rows_skipping_multiple": int(np.count_nonzero(fulfilled > 1)),
            "missing_next_goal": int(np.count_nonzero(~np.isfinite(corrected["next_goal_distance_m"]))),
        }), flush=True)
        completed.append(metadata["data_file"])
        _write_json(progress_path, {
            "completed_episode_count": len(completed),
            "expected_episode_count": len(sidecars),
            "completed_source_shards": completed,
            "updated_signature": signature,
        })
    eligible_rows = 0
    skipped_goals = 0
    multiple_skips = 0
    missing_next_goal = 0
    for sidecar_path in sorted((output_records / "goal_distance_corrections").glob("*/episode_*.json")):
        shard_path = sidecar_path.with_suffix(".npz")
        with np.load(shard_path, allow_pickle=False) as archive:
            fulfilled_counts = archive["fulfilled_goal_count"]
            distances = archive["next_goal_distance_m"]
            eligible_rows += int(fulfilled_counts.size)
            skipped_goals += int(np.count_nonzero(fulfilled_counts > 0))
            multiple_skips += int(np.count_nonzero(fulfilled_counts > 1))
            missing_next_goal += int(np.count_nonzero(~np.isfinite(distances)))
    summary = {
        "complete": True,
        "episode_count": len(sidecars),
        "eligible_agent_frame_count": eligible_rows,
        "frames_by_map": expected_frames_by_map,
        "agent_frames_skipping_at_least_one_goal": skipped_goals,
        "agent_frames_skipping_multiple_goals": multiple_skips,
        "agent_frames_without_a_visible_unfulfilled_goal": missing_next_goal,
        "algorithm": CORRECTION_ALGORITHM,
        "source_geometry_signature": source_manifest["signature"],
        "correction_manifest_signature": signature,
    }
    summary["comparison_to_original_goal_distance"] = _goal_distance_comparison(
        source_records, output_records / "goal_distance_corrections"
    )
    _write_json(output_records / "goal_distance_corrections" / "summary.json", summary)
    return manifest_path


def build_corrected_selection(source_model_dir: Path, output_model_dir: Path, *, per_bin: int, seed: int):
    source_model_dir = model_directory(source_model_dir)
    output_model_dir = model_directory(output_model_dir)
    source_records = records_directory(source_model_dir)
    output_records = records_directory(output_model_dir)
    pool = selector._load_candidate_pool(source_records)
    correction_records = output_records / "goal_distance_corrections"
    correction_manifest = json.loads((correction_records / "manifest.json").read_text(encoding="utf-8"))
    rows = pool["rows"]
    corrected_values = np.full(rows["frame_idx"].shape, np.nan, dtype=np.float32)
    correction_arrays = {
        name: np.full(rows["frame_idx"].shape, fill, dtype=dtype)
        for name, dtype, fill in (
            ("source_nearest_visible_goal_distance_m", np.float32, np.nan),
            ("resolved_goal_idx", np.int32, -1),
            ("fulfilled_goal_count", np.int32, 0),
            ("goal_radius_m", np.float32, np.nan),
            ("current_goal_distance_m", np.float32, np.nan),
        )
    }
    for shard_index, shard in enumerate(pool["shards"]):
        correction_path = correction_records / shard["map_name"] / Path(shard["source_shard"]).name
        if not correction_path.is_file():
            raise FileNotFoundError(f"Corrected source geometry is missing: {correction_path}")
        sidecar = json.loads(correction_path.with_suffix(".json").read_text(encoding="utf-8"))
        if sidecar["source_geometry_signature"] != pool["source_geometry_manifest_signature"]:
            raise ValueError(f"Correction was generated from a different source run: {correction_path}")
        with np.load(correction_path, allow_pickle=False) as archive:
            overlay = {name: archive[name] for name in archive.files}
        candidate_rows = np.flatnonzero(rows["source_shard_index"] == shard_index)
        if candidate_rows.size != overlay["frame_idx"].size:
            raise ValueError(f"Eligible row count changed in corrected geometry: {correction_path}")
        for field in ("frame_idx", "agent_id", "slot", "current_goal_idx", "goal_count"):
            if not np.array_equal(overlay[field], rows[field][candidate_rows]):
                raise ValueError(f"Corrected geometry identity mismatch in field {field}: {correction_path}")
        corrected_values[candidate_rows] = overlay["next_goal_distance_m"]
        correction_arrays["source_nearest_visible_goal_distance_m"][candidate_rows] = rows["nearest_visible_goal_distance_m"][candidate_rows]
        correction_arrays["resolved_goal_idx"][candidate_rows] = overlay["resolved_goal_idx"]
        correction_arrays["fulfilled_goal_count"][candidate_rows] = overlay["fulfilled_goal_count"]
        correction_arrays["goal_radius_m"][candidate_rows] = overlay["goal_radius_m"]
        correction_arrays["current_goal_distance_m"][candidate_rows] = rows["current_goal_distance_m"][candidate_rows]

    rows["nearest_visible_goal_distance_m"] = corrected_values
    pool["bin_indices"]["nearest_visible_goal_distance_m"] = selector._axis_bin_index(
        corrected_values, selector.PRIMARY_AXES["nearest_visible_goal_distance_m"]
    )
    source_relative = source_records.resolve().relative_to(selector.PROJECT_DIR.resolve()).as_posix()
    selection = selector.build_selection(
        pool, model=output_model_dir.name, source_records_dir=source_relative,
        seed=seed, per_bin=per_bin,
    )
    selection["algorithm"] = CORRECTION_ALGORITHM
    selection["corrected_geometry_manifest_signature"] = correction_manifest["signature"]
    selection["requested_per_bin"] = per_bin
    selection["requested_state_count"] = per_bin * sum(len(edges) - 1 for edges in selector.PRIMARY_AXES.values())
    selection["selected_state_count"] = len(selection["states"])
    correction_row_by_identity = {
        (int(rows["map_index"][index]), int(rows["episode_index"][index]), int(rows["frame_idx"][index]),
         int(rows["agent_id"][index]), int(rows["slot"][index])): index
        for index in range(rows["frame_idx"].size)
    }
    for state in selection["states"]:
        identity = (
            int(state["map_index"]), int(state["episode_index"]), int(state["frame_idx"]),
            int(state["agent_id"]), int(state["slot"]),
        )
        index = correction_row_by_identity[identity]
        state["distance_corrections_m"] = {
            "nearest_visible_goal_distance_m": None if not np.isfinite(corrected_values[index]) else float(corrected_values[index]),
            "source_nearest_visible_goal_distance_m": None if not np.isfinite(correction_arrays["source_nearest_visible_goal_distance_m"][index]) else float(correction_arrays["source_nearest_visible_goal_distance_m"][index]),
            "current_goal_idx": int(state["current_goal_idx"]),
            "resolved_goal_idx": int(correction_arrays["resolved_goal_idx"][index]),
            "fulfilled_goal_count": int(correction_arrays["fulfilled_goal_count"][index]),
            "goal_radius_m": float(correction_arrays["goal_radius_m"][index]),
            "goal_hit_rule": CORRECTION_ALGORITHM,
        }
        state["distances_m"]["nearest_visible_goal_distance_m"] = state["distance_corrections_m"]["nearest_visible_goal_distance_m"]
    correction_signature = correction_manifest["signature"]
    selection["signature"] = _canonical_sha256({
        "base_signature": selection["signature"],
        "corrected_geometry_signature": correction_signature,
        "algorithm": CORRECTION_ALGORITHM,
        "requested_per_bin": per_bin,
    })
    selection_path = output_records / "selection.json"
    selector._write_selection(selection_path, selection)
    return selection


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    if args.per_bin <= 0 or args.seed < 0:
        raise ValueError("--per-bin must be positive and --seed must be non-negative")
    source_model_dir = model_directory(args.source_model_dir)
    output_model_dir = model_directory(args.model_dir)
    if source_model_dir.name != output_model_dir.name:
        raise ValueError("source and output model names must match")
    collect_corrected_references(source_model_dir, output_model_dir)
    selection = build_corrected_selection(source_model_dir, output_model_dir, per_bin=args.per_bin, seed=args.seed)
    print(json.dumps({
        "model": output_model_dir.name,
        "selected_state_count": selection["selected_state_count"],
        "requested_state_count": selection["requested_state_count"],
        "shortages": selection["shortages"],
        "bin_summary": selection["bin_summary"],
        "selection": str(records_directory(output_model_dir) / "selection.json"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
