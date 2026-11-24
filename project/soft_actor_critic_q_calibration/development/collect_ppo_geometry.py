"""Collect PPO 4B self-play geometry references for every CARLA map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import collect_geometry as geometry
from .run_layout import initialize_model_directory, model_directory, records_directory
from .run_probes import sha256_file, stable_seed


DEFAULT_CHECKPOINT = (
    "project/soft_actor_critic_4b_runs_analysis/checkpoints/ppo/final_model.pt"
)
DEFAULT_CONFIG = (
    "project/soft_actor_critic_4b_runs_analysis/checkpoints/ppo/config.yaml"
)
DEFAULT_OUTPUT = "project/soft_actor_critic_q_calibration/runs/second_20261122/ppo"
DEFAULT_SEED = 20261122
SOURCE_MODEL = "baseline_ppo_4b_carla_2026-11-06_13-58-32_seed0"
CALIBRATION_ENV_OVERRIDES = {
    "capture_final_observations": True,
    "num_agents": 800,
    "partner_blindness_prob": 0.0,
    "partner_blindness_trigger_prob": 0.0,
    "phantom_braking_prob": 0.0,
    "phantom_braking_trigger_prob": 0.0,
}


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--steps-per-map", type=int, default=2560)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--map-index", type=int)
    return parser


def _source_hashes():
    paths = (
        Path(__file__),
        Path(geometry.__file__),
        Path(geometry.__file__).parent / "geometry_batch.py",
        Path(geometry.__file__).parent / "models.py",
        Path(geometry.__file__).parent / "ppo_models.py",
        Path(geometry.__file__).parent / "probe_env.py",
        Path(geometry.__file__).parent / "run_probes.py",
        Path(geometry.__file__).parent / "initial_geometry.py",
        Path(geometry.__file__).parent / "native_probe" / "build_private_binding.py",
        Path(geometry.__file__).parent / "native_probe" / "probe_export.h",
    )
    return {str(path): sha256_file(path) for path in paths}


def run_collection(args):
    import torch
    import yaml

    if args.steps_per_map <= 0:
        raise ValueError("--steps-per-map must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("PPO checkpoint and saved config must both exist")
    checkpoint_hash = sha256_file(checkpoint_path)
    source_env_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["env"]
    if not isinstance(source_env_config, dict):
        raise ValueError(f"saved PPO config lacks an env mapping: {config_path}")
    source_env_config = dict(source_env_config)
    repository_root = Path(__file__).resolve().parents[3]
    map_dir = Path(source_env_config["map_dir"]).expanduser()
    if not map_dir.is_absolute():
        map_dir = repository_root / map_dir
    source_env_config["map_dir"] = str(map_dir.resolve())
    from . import probe_env

    source_env_resolved_config = probe_env._as_drive_kwargs(source_env_config)
    source_env_resolved_config["map_dir"] = str(map_dir.resolve())
    env_config = dict(source_env_resolved_config)
    env_config.update(CALIBRATION_ENV_OVERRIDES)
    if env_config.get("simulation_mode") != "gigaflow" or int(env_config.get("num_maps", 0)) != 8:
        raise ValueError("PPO geometry collection requires the calibrated 8-map gigaflow config")
    if env_config.get("action_type") != "continuous" or env_config.get("dynamics_model") != "jerk":
        raise ValueError("PPO checkpoint must control the twelve jerk actions used by the SAC probes")
    torch.set_num_threads(2)

    output_model_dir = model_directory(args.model_dir)
    model_dir = initialize_model_directory(output_model_dir)
    records_dir = records_directory(model_dir)
    references_dir = records_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    readme_path = references_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# PPO geometry references\n\n"
            "Per-map NPZ shards store eligible pre-action agent-frame distances, joint sampled "
            "actions, and replay fingerprints. JSON sidecars record the scene seeds and shard hashes.\n"
        )

    maps = geometry._enumerate_maps(env_config, repository_root)
    if args.map_index is not None and not 0 <= args.map_index < len(maps):
        raise ValueError(f"--map-index must be in [0, {len(maps)})")
    selected_maps = maps if args.map_index is None else [maps[args.map_index]]
    scenario_length = int(env_config.get("scenario_length", 0))
    if scenario_length <= 0:
        raise ValueError("saved PPO config must have a positive scenario_length")

    from .ppo_models import FrozenPPO

    native_audit = geometry._native_audit(probe_env)
    adjacent_config_path = checkpoint_path.with_name("config.yaml")
    if not adjacent_config_path.is_file():
        raise FileNotFoundError(f"FrozenPPO requires config.yaml beside checkpoint: {adjacent_config_path}")
    if sha256_file(adjacent_config_path) != sha256_file(config_path):
        raise ValueError("Requested PPO config differs from the checkpoint-adjacent config FrozenPPO loads")

    # The model construction template stays fixed even for a map-index worker,
    # so every map shard records the same model-template layout identity.
    first_map = maps[0]
    model_template, _, model_layout_seed, _ = geometry._map_layout_template(
        probe_env, env_config, first_map["path"], args.seed, 0
    )
    try:
        ppo_model = FrozenPPO(checkpoint_path, model_template.model_env, device=args.device)
    finally:
        model_template.close()
    if ppo_model.audit.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Loaded PPO checkpoint hash differs from the requested snapshot")
    if not np.allclose(ppo_model.action_table, probe_env._jerk_action_table(), rtol=0.0, atol=1e-7):
        raise ValueError("PPO action table does not match the native twelve-action jerk table")
    train_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["train"]
    manifest_settings = {
        "source_model_dir": SOURCE_MODEL,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_checkpoint_config": str(config_path),
        "source_checkpoint_config_sha256": sha256_file(config_path),
        "source_env_config": source_env_config,
        "source_env_resolved_config": source_env_resolved_config,
        "experiment_env_overrides": CALIBRATION_ENV_OVERRIDES,
        "source_checkpoint_audit": {
            "model_hashes": ppo_model.initial_hashes,
            "gamma": float(train_config["gamma"]),
            "gae_lambda": float(train_config["gae_lambda"]),
            "vtrace_rho_clip": float(train_config["vtrace_rho_clip"]),
            "vtrace_c_clip": float(train_config["vtrace_c_clip"]),
            "use_value_bootstrapping": bool(train_config["use_value_bootstrapping"]),
        },
        "resolved_env_config": env_config,
        "maps": maps,
        "steps_per_map": int(args.steps_per_map),
        "unit": "native joint environment frames",
        "seed": int(args.seed),
        "device": str(args.device),
        "layout_sampling": {
            "distribution": "SAC PQL calibrated env (800-agent pool, erratic probabilities disabled) applied to PPO source config through probe_env.create_template",
            "coverage_retry_limit": geometry.MAX_LAYOUT_DRAWS,
            "retry_seed": "stable_seed(seed, geometry-layout, map_name, episode_index, draw_index)",
            "model_template_layout_seed": int(model_layout_seed),
        },
        "scene_seed": "stable_seed(seed, geometry-scene, map_name, episode_index)",
        "policy_action_rng": "stable_seed(seed, geometry-policy, map_name, episode_index), categorical per slot",
        "geometry_fields": list(geometry.GEOMETRY_FIELDS),
        "geometry_row_selection": "eligible pre-action frame-slot pairs only",
        "native_build": native_audit,
        "source_hashes": _source_hashes(),
    }
    manifest = geometry._open_or_create_geometry_manifest(
        records_dir / "geometry_manifest.json", geometry.sanitize_json(manifest_settings)
    )
    run_signature = manifest["signature"]
    models = ppo_model

    for map_record in selected_maps:
        completed, cumulative_frames = geometry._completed_map_frames(
            references_dir, map_record, run_signature
        )
        geometry._update_map_progress(
            references_dir, map_record, run_signature, args.steps_per_map, completed, cumulative_frames
        )
        episode_index = len(completed)
        while cumulative_frames < args.steps_per_map:
            template, agent_count, layout_seed, draw_index = geometry._map_layout_template(
                probe_env, env_config, map_record["path"], args.seed, episode_index
            )
            try:
                scene_seed = stable_seed(args.seed, "geometry-scene", map_record["map_name"], episode_index)
                policy_seed = stable_seed(args.seed, "geometry-policy", map_record["map_name"], episode_index)
                frame_limit = min(scenario_length, args.steps_per_map - cumulative_frames)
                episode = geometry._collect_episode(
                    probe_env.SceneBatch, template, models, map_record["path"],
                    agent_count, scene_seed, policy_seed, frame_limit,
                )
            finally:
                template.close()
            geometry._save_episode(
                references_dir, map_record, episode_index, cumulative_frames, run_signature,
                layout_seed, draw_index, scene_seed, models.action_table, episode,
            )
            cumulative_frames += episode["frame_count"]
            completed, counted_frames = geometry._completed_map_frames(
                references_dir, map_record, run_signature
            )
            if counted_frames != cumulative_frames:
                raise RuntimeError("Saved PPO episode frame count differs from its metadata")
            geometry._update_map_progress(
                references_dir, map_record, run_signature, args.steps_per_map, completed, cumulative_frames
            )
            print(json.dumps({
                "map": map_record["map_name"],
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
            models.verify_frozen()
            if episode["frame_count"] <= 0:
                raise RuntimeError("PPO geometry episode made no frame-count progress")
        print(json.dumps({
            "map": map_record["map_name"],
            "status": "complete",
            "cumulative_joint_frames": cumulative_frames,
            "target_joint_frames": args.steps_per_map,
            "completed_episodes": len(completed),
        }), flush=True)
    models.verify_frozen()


def main(argv=None):
    run_collection(build_argument_parser().parse_args(argv))


if __name__ == "__main__":
    main()
