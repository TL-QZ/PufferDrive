"""Collect frozen-policy, self-play geometry references for every Drive map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .run_layout import initialize_model_directory, model_directory, records_directory
from .run_probes import (
    atomic_write_json,
    atomic_write_npz,
    current_utc,
    observation_fingerprint,
    sample_actions,
    sanitize_json,
    sha256_file,
    stable_seed,
)


DEFAULT_SEED = 20261122
MAX_LAYOUT_DRAWS = 128
GEOMETRY_FIELDS = (
    "nearest_observed_partner_center_distance_m",
    "nearest_observed_road_boundary_distance_m",
    "nearest_visible_goal_distance_m",
    "current_goal_distance_m",
)
ARRAY_FORMAT_VERSION = 1


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Output model directory under runs/<run>/<model>")
    parser.add_argument("--source-model-dir", type=Path, required=True,
                        help="First-run PQL or TES model directory holding records/manifest.json")
    parser.add_argument("--steps-per-map", type=int, required=True,
                        help="Cumulative native joint frames to collect for each selected map")
    parser.add_argument("--device", required=True,
                        help="Torch inference device, fixed for the collection run (for example cuda:0)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Shared deterministic seed for both checkpoints (default: {DEFAULT_SEED})")
    parser.add_argument("--map-index", type=int,
                        help="Collect only this map index from the explicit sorted map list")
    return parser


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(sanitize_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _open_or_create_geometry_manifest(path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    signature = _canonical_sha256(settings)
    expected = {"format_version": ARRAY_FORMAT_VERSION, "signature": signature, "settings": settings}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature or existing.get("settings") != sanitize_json(settings):
            raise ValueError(
                f"Geometry manifest does not match requested inputs at {path}; choose a new model-dir"
            )
        return existing
    atomic_write_json(path, {**expected, "created_at": current_utc()})
    return {**expected, "created_at": current_utc()}


def _enumerate_maps(env_config: dict[str, Any], repository_root: Path) -> list[dict[str, Any]]:
    map_dir = Path(env_config["map_dir"]).expanduser()
    if not map_dir.is_absolute():
        map_dir = repository_root / map_dir
    map_dir = map_dir.resolve()
    map_paths = sorted(path.resolve() for path in map_dir.glob("*.bin"))
    map_count = int(env_config["num_maps"])
    if len(map_paths) < map_count:
        raise FileNotFoundError(
            f"Training config requests {map_count} maps but only {len(map_paths)} .bin files exist in {map_dir}"
        )
    map_paths = map_paths[:map_count]
    if len(map_paths) != 8:
        raise ValueError(f"Expected the first-run training config to enumerate 8 CARLA maps, got {len(map_paths)}")
    return [
        {
            "map_index": map_index,
            "map_name": path.stem.removeprefix("opendrive__"),
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for map_index, path in enumerate(map_paths)
    ]


def _load_source(source_model_dir: Path, repository_root: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = records_directory(source_model_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"First-run model manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_record = manifest.get("checkpoint")
    resolved_env = manifest.get("resolved_config", {}).get("resolved_env")
    if not isinstance(checkpoint_record, dict) or not isinstance(resolved_env, dict):
        raise ValueError(f"First-run manifest lacks checkpoint or resolved environment: {manifest_path}")
    checkpoint_path = Path(checkpoint_record["path"]).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = repository_root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"First-run checkpoint not found: {checkpoint_path}")
    actual_checkpoint_hash = sha256_file(checkpoint_path)
    expected_checkpoint_hash = checkpoint_record.get("sha256")
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise ValueError(
            f"Checkpoint hash differs from first-run manifest: expected {expected_checkpoint_hash}, "
            f"got {actual_checkpoint_hash} for {checkpoint_path}"
        )

    env_config = dict(resolved_env)
    map_dir = Path(env_config["map_dir"]).expanduser()
    if not map_dir.is_absolute():
        map_dir = repository_root / map_dir
    env_config["map_dir"] = str(map_dir.resolve())
    env_config.update(manifest.get("experiment_env_overrides", {}))
    if env_config.get("simulation_mode") != "gigaflow":
        raise ValueError("Geometry collection requires the first-run gigaflow training distribution")
    if int(env_config.get("num_maps", 0)) != 8:
        raise ValueError("Geometry collection requires exactly eight configured CARLA maps")
    return manifest, checkpoint_path, env_config


def _source_hashes() -> dict[str, str]:
    development_dir = Path(__file__).resolve().parent
    paths = (
        Path(__file__),
        development_dir / "geometry_batch.py",
        development_dir / "models.py",
        development_dir / "probe_env.py",
        development_dir / "run_probes.py",
        development_dir / "initial_geometry.py",
        development_dir / "native_probe" / "build_private_binding.py",
        development_dir / "native_probe" / "probe_export.h",
    )
    return {str(path): sha256_file(path) for path in paths}


def _native_audit(probe_env) -> dict[str, str]:
    module = probe_env.ensure_private_binding()
    return {
        "source_fingerprint": probe_env.native_build_fingerprint(),
        "private_binding_path": str(Path(module.__file__).resolve()),
        "private_binding_sha256": sha256_file(module.__file__),
    }


def _map_layout_template(probe_env, env_config, map_path, base_seed, episode_index):
    map_path = str(Path(map_path).resolve())
    map_name = Path(map_path).stem.removeprefix("opendrive__")
    for layout_draw_index in range(MAX_LAYOUT_DRAWS):
        layout_seed = stable_seed(
            base_seed, "geometry-layout", map_name, int(episode_index), layout_draw_index
        )
        template = probe_env.create_template(env_config, seed=layout_seed)
        agent_count = template.map_agent_counts.get(map_path)
        if agent_count is not None and int(agent_count) > 0:
            return template, int(agent_count), layout_seed, layout_draw_index
        template.close()
    raise RuntimeError(
        f"Training-layout sampling did not cover {map_name} in {MAX_LAYOUT_DRAWS} deterministic draws "
        f"for episode {episode_index}"
    )


def _assemble_episode_arrays(
    *,
    agent_ids,
    eligibility,
    current_goal_indices,
    goal_counts,
    geometry_series,
    joint_actions,
    fingerprints,
    observation_fingerprints,
):
    agent_ids = np.asarray(agent_ids, dtype=np.int32)
    eligibility = np.asarray(eligibility, dtype=np.bool_)
    current_goal_indices = np.asarray(current_goal_indices, dtype=np.int32)
    goal_counts = np.asarray(goal_counts, dtype=np.int32)
    joint_actions = np.asarray(joint_actions, dtype=np.int32)
    fingerprints = np.asarray(fingerprints, dtype="U64")
    observation_fingerprints = np.asarray(observation_fingerprints, dtype="U64")
    if agent_ids.ndim != 2 or eligibility.shape != agent_ids.shape:
        raise ValueError("agent_ids and eligibility must have matching [frames, agents] shapes")
    frame_count, agent_count = agent_ids.shape
    if current_goal_indices.shape != agent_ids.shape or goal_counts.shape != agent_ids.shape:
        raise ValueError("goal metadata must match the [frames, agents] scene shape")
    if joint_actions.shape != agent_ids.shape:
        raise ValueError("joint_actions must have shape [frames, agents]")
    if fingerprints.shape != (frame_count,) or observation_fingerprints.shape != (frame_count,):
        raise ValueError("fingerprints must contain one value per pre-action frame")
    if np.any(eligibility & (agent_ids < 0)):
        raise ValueError("an eligible slot cannot have a negative native agent_id")

    frame_indices, slots = np.nonzero(eligibility)
    result = {
        "frame_idx": frame_indices.astype(np.int32),
        "agent_id": agent_ids[frame_indices, slots],
        "slot": slots.astype(np.int32),
        "current_goal_idx": current_goal_indices[frame_indices, slots],
        "goal_count": goal_counts[frame_indices, slots],
        "eligible": np.ones(frame_indices.size, dtype=np.bool_),
        "preaction_eligible": eligibility,
        "joint_actions": joint_actions,
        "fingerprints": fingerprints,
        "observation_fingerprints": observation_fingerprints,
    }
    for name in GEOMETRY_FIELDS:
        distances = np.asarray(geometry_series[name], dtype=np.float32)
        if distances.shape != agent_ids.shape:
            raise ValueError(f"{name} must have shape {agent_ids.shape}, got {distances.shape}")
        if np.any(np.isinf(distances[eligibility])):
            raise ValueError(f"{name} contains infinity for an eligible row")
        result[name] = distances[frame_indices, slots]
    return result


def _collect_episode(
    scene_batch_type,
    template,
    models,
    map_path,
    agent_count,
    scene_seed,
    policy_seed,
    frame_limit,
):
    from .geometry_batch import geometry_batch

    batch = scene_batch_type.from_template(
        template, map_path, agent_count, copies=1, scene_seed=scene_seed
    )
    rng = np.random.default_rng(policy_seed)
    agent_ids_series = []
    eligibility_series = []
    goal_indices_series = []
    goal_counts_series = []
    geometry_series = {name: [] for name in GEOMETRY_FIELDS}
    joint_actions = []
    fingerprints = []
    observation_fingerprints = []
    episode_seed = None
    boundary_captured = False
    try:
        batch.reset()
        for _ in range(frame_limit):
            metadata = batch.metadata()
            observations = np.asarray(batch.observations[0], dtype=np.float32)
            if observations.shape[0] != agent_count:
                raise RuntimeError("native scene observation count differs from sampled training layout")
            agent_ids = np.asarray(metadata["agent_id"][0], dtype=np.int32)
            eligible = np.asarray(metadata["eligible"][0], dtype=np.bool_)
            current_goal_idx = np.asarray(metadata["current_goal_idx"][0], dtype=np.int32)
            goal_count = np.asarray(metadata["goal_count"][0], dtype=np.int32)
            if any(value.shape != (agent_count,) for value in (agent_ids, eligible, current_goal_idx, goal_count)):
                raise RuntimeError("native scene metadata does not match sampled agent count")
            if episode_seed is None:
                episode_seed = int(metadata["episode_seed"][0])
            elif episode_seed != int(metadata["episode_seed"][0]):
                raise RuntimeError("native episode seed changed inside one reference episode")

            geometry = {
                name: np.full(agent_count, np.nan, dtype=np.float32)
                for name in GEOMETRY_FIELDS
            }
            if np.any(eligible):
                eligible_geometry = geometry_batch(
                    observations[eligible],
                    template.probe_config,
                    current_goal_idx=current_goal_idx[eligible],
                    goal_count=goal_count[eligible],
                )
                for name in GEOMETRY_FIELDS:
                    geometry[name][eligible] = eligible_geometry[name]
            for name in GEOMETRY_FIELDS:
                geometry_series[name].append(geometry[name])
            agent_ids_series.append(agent_ids.copy())
            eligibility_series.append(eligible.copy())
            goal_indices_series.append(current_goal_idx.copy())
            goal_counts_series.append(goal_count.copy())
            fingerprints.append(batch.fingerprint(0))
            observation_fingerprints.append(observation_fingerprint(observations))

            probabilities, _ = models.policy(observations)
            actions = sample_actions(probabilities, rng)
            joint_actions.append(actions)
            step_result = batch.step(models.action_table[actions][None, ...])
            if bool(step_result["metadata"]["scene_done"][0]):
                boundary_captured = True
                break
    finally:
        batch.close()

    if not joint_actions:
        raise RuntimeError(f"Native scene for {map_path} produced no action frames")
    arrays = _assemble_episode_arrays(
        agent_ids=np.stack(agent_ids_series),
        eligibility=np.stack(eligibility_series),
        current_goal_indices=np.stack(goal_indices_series),
        goal_counts=np.stack(goal_counts_series),
        geometry_series={name: np.stack(values) for name, values in geometry_series.items()},
        joint_actions=np.stack(joint_actions),
        fingerprints=fingerprints,
        observation_fingerprints=observation_fingerprints,
    )
    return {
        "arrays": arrays,
        "frame_count": len(joint_actions),
        "agent_count": int(agent_count),
        "episode_seed": int(episode_seed),
        "boundary_captured": boundary_captured,
        "frame_limit_reached": not boundary_captured,
    }


def _episode_paths(references_dir: Path, map_name: str, episode_index: int) -> tuple[Path, Path]:
    map_dir = references_dir / map_name
    stem = f"episode_{episode_index:05d}"
    return map_dir / f"{stem}.json", map_dir / f"{stem}.npz"


def _save_episode(references_dir, map_record, episode_index, cumulative_start, run_signature,
                  layout_seed, layout_draw_index, scene_seed, action_table, episode):
    metadata_path, data_path = _episode_paths(references_dir, map_record["map_name"], episode_index)
    arrays = episode["arrays"]
    atomic_write_npz(data_path, **arrays)
    metadata = {
        "format_version": ARRAY_FORMAT_VERSION,
        "run_signature": run_signature,
        "map_index": map_record["map_index"],
        "map_name": map_record["map_name"],
        "map_path": map_record["path"],
        "map_sha256": map_record["sha256"],
        "episode_index": int(episode_index),
        "layout_seed": int(layout_seed),
        "layout_draw_index": int(layout_draw_index),
        "scene_seed": int(scene_seed),
        "episode_seed": int(episode["episode_seed"]),
        "agent_count": int(episode["agent_count"]),
        "frame_count": int(episode["frame_count"]),
        "eligible_row_count": int(arrays["frame_idx"].size),
        "cumulative_frame_start": int(cumulative_start),
        "cumulative_frame_end": int(cumulative_start + episode["frame_count"]),
        "boundary_captured": bool(episode["boundary_captured"]),
        "frame_limit_reached": bool(episode["frame_limit_reached"]),
        "action_table_normalized": np.asarray(action_table, dtype=np.float32).tolist(),
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "completed_at": current_utc(),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def _load_completed_episode(references_dir, map_record, episode_index, run_signature):
    metadata_path, data_path = _episode_paths(references_dir, map_record["map_name"], episode_index)
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not data_path.is_file():
        raise ValueError(f"Episode metadata exists without its NPZ shard: {metadata_path}")
    expected = {
        "run_signature": run_signature,
        "map_index": map_record["map_index"],
        "map_name": map_record["map_name"],
        "map_path": map_record["path"],
        "map_sha256": map_record["sha256"],
        "episode_index": episode_index,
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
    }
    mismatches = [name for name, value in expected.items() if metadata.get(name) != value]
    if mismatches:
        raise ValueError(f"Existing episode shard is incompatible or corrupt ({mismatches}): {data_path}")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
        required = {
            "frame_idx", "agent_id", "slot", "current_goal_idx", "goal_count", "eligible",
            "preaction_eligible", "joint_actions", "fingerprints", "observation_fingerprints",
            *GEOMETRY_FIELDS,
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"Episode shard is missing required arrays {missing}: {data_path}")
        frame_count = int(metadata["frame_count"])
        joint_actions = arrays["joint_actions"]
        eligibility = arrays["preaction_eligible"]
        if joint_actions.ndim != 2 or joint_actions.shape[0] != frame_count:
            raise ValueError(f"Episode shard joint_actions has invalid shape: {data_path}")
        if eligibility.shape != joint_actions.shape or eligibility.dtype != np.bool_:
            raise ValueError(f"Episode shard preaction_eligible has invalid shape or dtype: {data_path}")
        if arrays["fingerprints"].shape != (frame_count,) or arrays["observation_fingerprints"].shape != (frame_count,):
            raise ValueError(f"Episode shard fingerprints have invalid shape: {data_path}")
        expected_frames, expected_slots = np.nonzero(eligibility)
        if not np.array_equal(arrays["frame_idx"], expected_frames) or not np.array_equal(
            arrays["slot"], expected_slots
        ):
            raise ValueError(f"Episode shard rows do not exactly cover eligible frame-slots: {data_path}")
        row_count = expected_frames.size
        for name in required - {
            "preaction_eligible", "joint_actions", "fingerprints", "observation_fingerprints"
        }:
            if arrays[name].shape != (row_count,):
                raise ValueError(f"Episode shard row field {name} has invalid shape: {data_path}")
        if int(metadata["eligible_row_count"]) != row_count:
            raise ValueError(f"Episode shard eligible row count differs from metadata: {data_path}")
    return metadata


def _completed_map_frames(references_dir, map_record, run_signature):
    map_dir = references_dir / map_record["map_name"]
    if not map_dir.exists():
        return [], 0
    completed = []
    frame_count = 0
    episode_index = 0
    while True:
        metadata = _load_completed_episode(references_dir, map_record, episode_index, run_signature)
        if metadata is None:
            orphan = _episode_paths(references_dir, map_record["map_name"], episode_index)[1]
            if orphan.exists():
                # NPZ is atomically written before its sidecar. An orphan is a safe retry target.
                break
            break
        if int(metadata["cumulative_frame_start"]) != frame_count:
            raise ValueError(f"Non-contiguous cumulative frame range in {map_dir}")
        frame_count += int(metadata["frame_count"])
        completed.append(metadata)
        episode_index += 1
    return completed, frame_count


def _update_map_progress(references_dir, map_record, run_signature, target_frames, completed, frame_count):
    map_dir = references_dir / map_record["map_name"]
    atomic_write_json(map_dir / "progress.json", {
        "run_signature": run_signature,
        "map_index": map_record["map_index"],
        "map_name": map_record["map_name"],
        "target_joint_frames": int(target_frames),
        "completed_joint_frames": int(frame_count),
        "completed_episodes": len(completed),
        "complete": frame_count >= target_frames,
        "updated_at": current_utc(),
    })


def run_collection(args) -> None:
    import torch
    import yaml

    development_dir = Path(__file__).resolve().parent
    repository_root = development_dir.parents[2]
    if args.steps_per_map <= 0:
        raise ValueError("--steps-per-map must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    torch.set_num_threads(2)

    output_model_dir = model_directory(args.model_dir)
    source_model_dir = model_directory(args.source_model_dir)
    if output_model_dir == source_model_dir:
        raise ValueError("--model-dir must be separate from --source-model-dir to preserve first-run records")

    source_manifest, checkpoint_path, env_config = _load_source(source_model_dir, repository_root)
    maps = _enumerate_maps(env_config, repository_root)
    if args.map_index is not None and not 0 <= args.map_index < len(maps):
        raise ValueError(f"--map-index must be in [0, {len(maps)})")
    selected_maps = maps if args.map_index is None else [maps[args.map_index]]
    scenario_length = int(env_config.get("scenario_length") or 0)
    if scenario_length <= 0:
        raise ValueError("resolved first-run environment must have a positive scenario_length")

    checkpoint_config_path = checkpoint_path.with_name("config.yaml")
    if not checkpoint_config_path.is_file():
        raise FileNotFoundError(f"checkpoint config not found: {checkpoint_config_path}")
    checkpoint_config = yaml.safe_load(checkpoint_config_path.read_text(encoding="utf-8"))
    if not isinstance(checkpoint_config, dict) or not isinstance(checkpoint_config.get("policy"), dict):
        raise ValueError(f"checkpoint config must contain a policy mapping: {checkpoint_config_path}")

    model_dir = initialize_model_directory(output_model_dir)
    records_dir = records_directory(model_dir)
    references_dir = records_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    readme_path = references_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# Geometry references\n\n"
            "Each map directory contains atomic per-episode NPZ shards and JSON sidecars. "
            "Distance rows include only eligible pre-action agent-frames; the full eligibility "
            "matrix, joint sampled actions, native-state fingerprints, and observation "
            "fingerprints are stored for replay and coverage audits.\n"
        )

    from . import probe_env
    native_audit = _native_audit(probe_env)
    manifest_settings = {
        "source_model_dir": str(source_model_dir),
        "source_manifest_signature": source_manifest.get("signature"),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_checkpoint_config_sha256": sha256_file(checkpoint_config_path),
        "source_checkpoint_audit": source_manifest.get("checkpoint", {}).get("audit", {}),
        "resolved_env_config": env_config,
        "maps": maps,
        "steps_per_map": int(args.steps_per_map),
        "unit": "native joint environment frames",
        "seed": int(args.seed),
        "device": str(args.device),
        "layout_sampling": {
            "distribution": "first-run gigaflow training layout from probe_env.create_template",
            "coverage_retry_limit": MAX_LAYOUT_DRAWS,
            "retry_seed": "stable_seed(seed, geometry-layout, map_name, episode_index, draw_index)",
        },
        "scene_seed": "stable_seed(seed, geometry-scene, map_name, episode_index), shared across checkpoints",
        "policy_action_rng": "stable_seed(seed, geometry-policy, map_name, episode_index), categorical per slot",
        "geometry_fields": list(GEOMETRY_FIELDS),
        "geometry_row_selection": "eligible pre-action frame-slot pairs only",
        "native_build": native_audit,
        "source_hashes": _source_hashes(),
    }
    geometry_manifest = _open_or_create_geometry_manifest(
        records_dir / "geometry_manifest.json", sanitize_json(manifest_settings)
    )
    run_signature = geometry_manifest["signature"]

    model_module = __import__(
        "project.soft_actor_critic_q_calibration.development.models", fromlist=["FrozenModels"]
    )
    models = None
    try:
        for map_record in selected_maps:
            completed, cumulative_frames = _completed_map_frames(
                references_dir, map_record, run_signature
            )
            _update_map_progress(
                references_dir, map_record, run_signature, args.steps_per_map,
                completed, cumulative_frames,
            )
            episode_index = len(completed)
            while cumulative_frames < args.steps_per_map:
                template, agent_count, layout_seed, draw_index = _map_layout_template(
                    probe_env, env_config, map_record["path"], args.seed, episode_index
                )
                try:
                    if models is None:
                        models = model_module.FrozenModels(
                            checkpoint_path, template.model_env, device=args.device, compile_actor=False
                        )
                        expected_hashes = source_manifest.get("checkpoint", {}).get("audit", {}).get("model_hashes")
                        if expected_hashes and expected_hashes != models.initial_hashes:
                            raise ValueError("Loaded frozen model hashes do not match the first-run manifest audit")
                        if models.audit["checkpoint_sha256"] != manifest_settings["source_checkpoint_sha256"]:
                            raise ValueError("Loaded checkpoint hash changed after source-manifest validation")
                        if not np.allclose(
                            models.action_table,
                            probe_env._jerk_action_table(),
                            rtol=0.0,
                            atol=1e-7,
                        ):
                            raise ValueError("Checkpoint action table differs from native Drive jerk action order")

                    map_name = map_record["map_name"]
                    scene_seed = stable_seed(args.seed, "geometry-scene", map_name, episode_index)
                    policy_seed = stable_seed(args.seed, "geometry-policy", map_name, episode_index)
                    frame_limit = min(scenario_length, args.steps_per_map - cumulative_frames)
                    episode = _collect_episode(
                        probe_env.SceneBatch,
                        template,
                        models,
                        map_record["path"],
                        agent_count,
                        scene_seed,
                        policy_seed,
                        frame_limit,
                    )
                finally:
                    template.close()
                _save_episode(
                    references_dir,
                    map_record,
                    episode_index,
                    cumulative_frames,
                    run_signature,
                    layout_seed,
                    draw_index,
                    scene_seed,
                    models.action_table,
                    episode,
                )
                cumulative_frames += episode["frame_count"]
                completed, counted_frames = _completed_map_frames(
                    references_dir, map_record, run_signature
                )
                if counted_frames != cumulative_frames:
                    raise RuntimeError("Saved episode frame count differs from contiguous shard metadata")
                _update_map_progress(
                    references_dir, map_record, run_signature, args.steps_per_map,
                    completed, cumulative_frames,
                )
                print(json.dumps({
                    "map": map_name,
                    "episode_index": episode_index,
                    "agent_count": agent_count,
                    "episode_frames": episode["frame_count"],
                    "cumulative_joint_frames": cumulative_frames,
                    "target_joint_frames": args.steps_per_map,
                    "eligible_geometry_rows": int(episode["arrays"]["frame_idx"].size),
                    "scene_seed": scene_seed,
                    "episode_seed": episode["episode_seed"],
                }), flush=True)
                episode_index += 1
                if models is not None:
                    models.verify_frozen()
                if episode["frame_count"] <= 0:
                    raise RuntimeError("An episode made no cumulative-frame progress")
            print(json.dumps({
                "map": map_record["map_name"],
                "status": "complete",
                "cumulative_joint_frames": cumulative_frames,
                "target_joint_frames": args.steps_per_map,
                "completed_episodes": len(completed),
            }), flush=True)
    finally:
        if models is not None:
            models.verify_frozen()


def main(argv=None) -> None:
    args = build_argument_parser().parse_args(argv)
    run_collection(args)


if __name__ == "__main__":
    main()
