"""Collect twelve-action SAC probes from states selected in geometry shards."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .run_layout import initialize_model_directory, model_directory, records_directory
from .run_probes import (
    NUM_ACTIONS,
    _assemble_probe_bundle,
    _native_build_fingerprint,
    _run_branch_batch,
    _scene_parameters,
    atomic_write_json,
    build_branch_tasks,
    current_utc,
    initial_geometry,
    open_or_create_manifest,
    probe_is_complete,
    probe_paths,
    sanitize_json,
    sha256_file,
    source_hashes,
    stable_seed,
)


PROBE_HORIZON = 2560
REPEATS = 1
FORMAT_VERSION = 1
GEOMETRY_FIELDS = (
    "nearest_observed_partner_center_distance_m",
    "nearest_observed_road_boundary_distance_m",
    "nearest_visible_goal_distance_m",
    "current_goal_distance_m",
)


def build_argument_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Destination model directory under runs/<run>/<model>")
    parser.add_argument("--source-model-dir", type=Path, required=True,
                        help="Model directory under runs/second_20261122/<model>")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    return parser


def _canonical_sha256(value):
    encoded = json.dumps(sanitize_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _worker_name(worker_index, workers):
    return f"worker_{worker_index:03d}_of_{workers:03d}"


@contextmanager
def _worker_lock(records_dir, worker_index, workers):
    lock_dir = Path(records_dir) / "workers"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{_worker_name(worker_index, workers)}.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"worker index {worker_index} is already active for {workers} workers") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _worker_progress(records_dir, worker_index, workers, **fields):
    progress_dir = Path(records_dir) / "workers"
    progress_dir.mkdir(parents=True, exist_ok=True)
    path = progress_dir / f"{_worker_name(worker_index, workers)}.json"
    current = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(fields)
    current["updated_at"] = current_utc()
    atomic_write_json(path, current)


def _load_geometry_source(source_model_dir, repository_root):
    source_records = records_directory(source_model_dir)
    manifest_path = source_records / "geometry_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source geometry manifest not found: {manifest_path}")
    geometry_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = geometry_manifest.get("settings")
    if not isinstance(settings, dict) or not geometry_manifest.get("signature"):
        raise ValueError(f"invalid source geometry manifest: {manifest_path}")
    checkpoint_path = Path(settings["source_checkpoint"]).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = repository_root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {checkpoint_path}")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != settings.get("source_checkpoint_sha256"):
        raise ValueError(f"source checkpoint hash differs from geometry manifest: {checkpoint_path}")
    env_config = settings.get("resolved_env_config")
    if not isinstance(env_config, dict):
        raise ValueError("source geometry manifest lacks resolved_env_config")
    env_config = dict(env_config)
    map_dir = Path(env_config["map_dir"]).expanduser()
    if not map_dir.is_absolute():
        map_dir = repository_root / map_dir
    env_config["map_dir"] = str(map_dir.resolve())
    if env_config.get("compute_eval_metrics") is not False:
        raise ValueError("geometry prefix source must have compute_eval_metrics=false")
    if int(env_config.get("scenario_length", 0)) <= 0:
        raise ValueError("source environment must have a positive scenario_length")
    return source_records, geometry_manifest, settings, checkpoint_path, env_config


def _load_selection(path, *, source_records, geometry_manifest, model_name):
    if not path.is_file():
        raise FileNotFoundError(f"binned probe selection not found: {path}")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("model", "").lower() != model_name.lower():
        raise ValueError(
            f"selection model {selection.get('model')!r} does not match destination {model_name!r}"
        )
    expected_signature = geometry_manifest["signature"]
    if selection.get("source_geometry_manifest_signature") != expected_signature:
        raise ValueError("selection was created from a different source geometry manifest")
    source_records_value = selection.get("source_records_dir")
    if source_records_value:
        selected_records = Path(source_records_value).expanduser()
        if not selected_records.is_absolute():
            from .run_layout import PROJECT_DIR
            selected_records = PROJECT_DIR / selected_records
        if selected_records.resolve() != Path(source_records).resolve():
            raise ValueError("selection source_records_dir does not match --source-model-dir")
    states = selection.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("selection must contain a non-empty states list")
    state_ids = [str(state.get("state_id", "")) for state in states]
    if any(not value for value in state_ids) or len(set(state_ids)) != len(state_ids):
        raise ValueError("selection state_id values must be non-empty and unique")
    for state in states:
        missing = sorted({
            "state_id", "map_name", "map_index", "episode_index", "frame_idx", "agent_id",
            "slot", "source_shard", "fingerprint", "observation_fingerprint",
        } - set(state))
        if missing:
            raise ValueError(f"selection state is missing required fields {missing}")
    return selection


def _resolve_shard_path(source_records, source_shard):
    shard_path = (Path(source_records) / str(source_shard)).resolve()
    try:
        shard_path.relative_to(Path(source_records).resolve())
    except ValueError as error:
        raise ValueError(f"source_shard escapes source records directory: {source_shard}") from error
    return shard_path


def _load_source_episode(source_records, geometry_signature, state):
    shard_path = _resolve_shard_path(source_records, state["source_shard"])
    metadata_path = shard_path.with_suffix(".json")
    if not shard_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"source geometry shard or sidecar is missing: {shard_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_signature") != geometry_signature:
        raise ValueError(f"source shard belongs to a different geometry run: {metadata_path}")
    if metadata.get("data_file") != shard_path.name or metadata.get("data_sha256") != sha256_file(shard_path):
        raise ValueError(f"source geometry shard hash or filename does not match its sidecar: {shard_path}")
    if metadata.get("map_name") != state["map_name"] or int(metadata.get("map_index", -1)) != int(state["map_index"]):
        raise ValueError(f"selection map identity does not match geometry shard: {shard_path}")
    if int(metadata.get("episode_index", -1)) != int(state["episode_index"]):
        raise ValueError(f"selection episode identity does not match geometry shard: {shard_path}")

    with np.load(shard_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    source_row = _source_row_for_state(state, metadata, arrays)
    reference = _reference_from_episode(metadata, arrays)
    return shard_path, metadata_path, metadata, arrays, source_row, reference


def _source_row_for_state(state, metadata, arrays):
    required = {
        "frame_idx", "agent_id", "slot", "current_goal_idx", "goal_count", "preaction_eligible",
        "joint_actions", "fingerprints", "observation_fingerprints", *GEOMETRY_FIELDS,
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"source geometry shard is missing arrays {missing}")

    frame_idx = int(state["frame_idx"])
    slot = int(state["slot"])
    frame_count = int(metadata["frame_count"])
    agent_count = int(metadata["agent_count"])
    if not 0 <= frame_idx < frame_count or not 0 <= slot < agent_count:
        raise ValueError(f"selected frame/slot is outside the source shard: {state['state_id']}")
    if arrays["joint_actions"].shape != (frame_count, agent_count):
        raise ValueError("joint_actions shape differs from source sidecar")
    if arrays["preaction_eligible"].shape != (frame_count, agent_count):
        raise ValueError("preaction_eligible shape differs from source sidecar")
    if not bool(arrays["preaction_eligible"][frame_idx, slot]):
        raise ValueError(f"selected source frame-slot is not eligible: {state['state_id']}")

    row_indices = np.flatnonzero(
        (arrays["frame_idx"] == frame_idx) & (arrays["slot"] == slot)
    )
    if row_indices.size != 1:
        raise ValueError(f"selected source row is not unique in geometry shard: {state['state_id']}")
    row_idx = int(row_indices[0])
    source_row = {
        "frame_idx": frame_idx,
        "slot": slot,
        "agent_id": int(arrays["agent_id"][row_idx]),
        "current_goal_idx": int(arrays["current_goal_idx"][row_idx]),
        "goal_count": int(arrays["goal_count"][row_idx]),
        "fingerprint": str(arrays["fingerprints"][frame_idx]),
        "observation_fingerprint": str(arrays["observation_fingerprints"][frame_idx]),
        "distances_m": {
            name: (float(arrays[name][row_idx]) if np.isfinite(arrays[name][row_idx]) else None)
            for name in GEOMETRY_FIELDS
        },
    }
    if source_row["agent_id"] != int(state["agent_id"]):
        raise ValueError(f"selected agent_id differs from source geometry row: {state['state_id']}")
    if source_row["fingerprint"] != str(state["fingerprint"]):
        raise ValueError(f"selected native fingerprint differs from source geometry row: {state['state_id']}")
    if source_row["observation_fingerprint"] != str(state["observation_fingerprint"]):
        raise ValueError(
            f"selected observation fingerprint differs from source geometry row: {state['state_id']}"
        )
    if "distances_m" in state:
        for name, expected in state["distances_m"].items():
            observed = source_row["distances_m"].get(name)
            if expected is None or observed is None:
                if expected is not None or observed is not None:
                    raise ValueError(f"selected distance nullness differs for {name}: {state['state_id']}")
            elif not np.isclose(float(expected), observed, rtol=1e-6, atol=1e-5):
                raise ValueError(f"selected distance differs for {name}: {state['state_id']}")

    for field in ("current_goal_idx", "goal_count"):
        if field in state and int(state[field]) != source_row[field]:
            raise ValueError(f"selected {field} differs from source geometry row: {state['state_id']}")
    if "map_path" in state and Path(state["map_path"]).resolve() != Path(metadata["map_path"]).resolve():
        raise ValueError(f"selected map_path differs from source geometry row: {state['state_id']}")
    source_metadata = state.get("source_metadata")
    if isinstance(source_metadata, dict):
        for field in ("layout_seed", "layout_draw_index", "scene_seed", "episode_seed", "agent_count"):
            if field in source_metadata and int(source_metadata[field]) != int(metadata[field]):
                raise ValueError(f"selected source_metadata {field} differs: {state['state_id']}")

    return source_row


def _reference_from_episode(metadata, arrays):
    frame_count = int(metadata["frame_count"])

    reference = {
        "joint_actions": arrays["joint_actions"],
        "fingerprints": arrays["fingerprints"].astype(str).tolist(),
        "observation_fingerprints": arrays["observation_fingerprints"].astype(str).tolist(),
        "metadata_series": {},
        "initial_metadata": {},
        "frame_count": frame_count,
        "boundary_captured": bool(metadata.get("boundary_captured", False)),
        "reference_cap_reached": bool(metadata.get("frame_limit_reached", False)),
    }
    return reference


def _verify_replayed_state(branch_records, state, source_row):
    actual_metadata = branch_records[0].get("replay_start_metadata")
    if not isinstance(actual_metadata, dict):
        raise RuntimeError("branch collection did not retain replayed native start metadata")
    if branch_records[0].get("replay_start_fingerprint") != source_row["fingerprint"]:
        raise AssertionError(f"native state fingerprint mismatch at selected state {state['state_id']}")
    if branch_records[0].get("replay_start_observation_fingerprint") != source_row["observation_fingerprint"]:
        raise AssertionError(f"observation fingerprint mismatch at selected state {state['state_id']}")
    slot = int(state["slot"])
    for name in ("agent_id", "current_goal_idx", "goal_count"):
        if name not in actual_metadata:
            raise RuntimeError(f"replayed start metadata omitted {name!r}")
        if int(np.asarray(actual_metadata[name])[slot]) != int(source_row[name]):
            raise AssertionError(f"replayed {name} differs at selected state {state['state_id']}")
    if not bool(np.asarray(actual_metadata["eligible"])[slot]):
        raise AssertionError(f"selected agent is not eligible after prefix replay: {state['state_id']}")


def _assigned_bin(state):
    value = state.get("assigned_bin")
    if value is not None:
        return value
    tags = state.get("bin_tags", [])
    if isinstance(tags, list) and tags:
        if len(tags) == 1:
            return dict(tags[0])
        return [dict(tag) for tag in tags]
    return "unassigned"


def _bin_label(assigned_bin):
    if isinstance(assigned_bin, dict):
        return str(assigned_bin.get("bin_id") or f"{assigned_bin.get('axis')}:{assigned_bin.get('bin_index')}")
    if isinstance(assigned_bin, list):
        return ";".join(_bin_label(item) for item in assigned_bin)
    return str(assigned_bin)


def _worker_assignment(states, worker_index, workers):
    if workers < 1 or not 0 <= worker_index < workers:
        raise ValueError("worker_index must be in [0, workers) and workers must be positive")
    return list(states[worker_index::workers])


def _update_collection_summary(records_dir, selection, run_signature):
    records_dir = Path(records_dir)
    lock_path = records_dir / ".summary.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            selected_states = selection["states"]
            selected_by_bin = Counter(_bin_label(_assigned_bin(state)) for state in selected_states)
            state_ids = {str(state["state_id"]) for state in selected_states}
            completed = []
            for metadata_path in sorted((records_dir / "probes").glob("*.json")):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if metadata.get("run_signature") != run_signature:
                    continue
                if str(metadata.get("state_id", "")) not in state_ids:
                    continue
                if metadata.get("status") != "complete":
                    continue
                completed.append(metadata)
            completed_by_bin = Counter(_bin_label(item.get("assigned_bin", "unassigned")) for item in completed)
            branch_count = sum(len(item.get("branches", [])) for item in completed)
            metric_keys = sorted({
                key for item in completed for branch in item.get("branches", [])
                for key in branch.get("metrics_mean", {})
            })
            summary = {
                "format_version": FORMAT_VERSION,
                "run_signature": run_signature,
                "selection_state_count": len(selected_states),
                "completed_state_count": len(completed),
                "selected_by_bin": dict(sorted(selected_by_bin.items())),
                "completed_by_bin": dict(sorted(completed_by_bin.items())),
                "action_count": NUM_ACTIONS,
                "repeats": REPEATS,
                "continuation_horizon": PROBE_HORIZON,
                "expected_branch_count": len(selected_states) * NUM_ACTIONS,
                "completed_branch_count": branch_count,
                "eval_metric_key_count": len(metric_keys),
                "eval_metric_keys": metric_keys,
                "complete": len(completed) == len(selected_states) and branch_count == len(selected_states) * NUM_ACTIONS,
                "updated_at": current_utc(),
            }
            atomic_write_json(records_dir / "collection_summary.json", summary)
            return summary
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _source_hashes(development_dir, repo_root):
    paths = [
        Path(__file__),
        development_dir / "run_probes.py",
        development_dir / "models.py",
        development_dir / "probe_returns.py",
        development_dir / "initial_geometry.py",
        development_dir / "probe_env.py",
        development_dir / "native_probe" / "build_private_binding.py",
        development_dir / "native_probe" / "probe_export.h",
        development_dir / "native_probe" / "probe_horizon.h",
        repo_root / "pufferlib" / "sac_returns.py",
        repo_root / "pufferlib" / "ocean" / "drive" / "drive.h",
        repo_root / "pufferlib" / "ocean" / "drive" / "binding.c",
        repo_root / "pufferlib" / "ocean" / "drive" / "constants.h",
        repo_root / "pufferlib" / "ocean" / "env_binding.h",
        repo_root / "pufferlib" / "ocean" / "torch_sac.py",
        repo_root / "pufferlib" / "ocean" / "torch.py",
        repo_root / "pufferlib" / "ocean" / "drive" / "drive.py",
    ]
    return source_hashes(paths)


def validate_args(args):
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("--worker-index must be in [0, workers)")


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    validate_args(args)
    return run_collection(args)


def run_collection(args):
    import torch
    import yaml

    development_dir = Path(__file__).resolve().parent
    repo_root = development_dir.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    torch.set_num_threads(2)

    output_model_dir = initialize_model_directory(model_directory(args.model_dir))
    source_model_dir = model_directory(args.source_model_dir)
    if output_model_dir == source_model_dir:
        raise ValueError("--model-dir must be separate from --source-model-dir")
    model_name = output_model_dir.name
    source_records, geometry_manifest, geometry_settings, checkpoint_path, env_config = _load_geometry_source(
        source_model_dir, repo_root
    )
    selection_path = records_directory(output_model_dir) / "selection.json"
    selection = _load_selection(
        selection_path, source_records=source_records, geometry_manifest=geometry_manifest,
        model_name=model_name,
    )
    seed = int(selection.get("seed", geometry_settings.get("seed", 20261122)))
    if seed < 0:
        raise ValueError("selection seed must be non-negative")
    selection_sha = sha256_file(selection_path)
    records_dir = records_directory(output_model_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    assigned_states = sorted(
        _worker_assignment(selection["states"], args.worker_index, args.workers),
        key=lambda state: (
            str(state["source_shard"]), int(state["frame_idx"]), int(state["slot"]), str(state["state_id"])
        ),
    )
    with _worker_lock(records_dir, args.worker_index, args.workers):
        _worker_progress(
            records_dir, args.worker_index, args.workers,
            status="initializing", worker_id=args.worker_index, worker_count=args.workers,
            assigned_state_count=len(assigned_states), completed_state_count=0,
        )
        from . import probe_env
        from .models import FrozenModels
        from .probe_returns import compute_probe_measurements

        private_binding = probe_env.ensure_private_binding()
        native_fingerprint = _native_build_fingerprint(
            private_binding, probe_env.native_build_fingerprint()
        )
        source_file_hashes = _source_hashes(development_dir, repo_root)
        map_records = geometry_settings.get("maps", [])
        map_hashes = {record["path"]: record["sha256"] for record in map_records}
        run_settings = {
            "source_model_dir": str(source_model_dir),
            "source_geometry_manifest_signature": geometry_manifest["signature"],
            "selection_file_sha256": selection_sha,
            "selection_seed": seed,
            "worker_count": int(args.workers),
            "action_count": NUM_ACTIONS,
            "repeats": REPEATS,
            "continuation_horizon": PROBE_HORIZON,
            "branch_batch_size": NUM_ACTIONS,
            "device": str(args.device),
            "continuation_seed_scope": "shared_across_forced_actions_for_each_state",
            "prefix_replay": "compact_native_joint_action_replay_with_endpoint_fingerprint",
            "eval_metric_collection": {
                "compute_eval_metrics_during_prefix": False,
                "compute_eval_metrics_after_prefix": True,
                "scope": "probing_agent_forced_action_to_branch_end",
            },
            "action_table_normalized": probe_env._jerk_action_table().tolist(),
        }
        checkpoint_config_path = checkpoint_path.with_name("config.yaml")
        if not checkpoint_config_path.is_file():
            raise FileNotFoundError(f"source checkpoint config not found: {checkpoint_config_path}")
        checkpoint_config = yaml.safe_load(checkpoint_config_path.read_text(encoding="utf-8"))
        manifest = open_or_create_manifest(records_dir, {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "config_path": str(checkpoint_config_path),
                "config_sha256": sha256_file(checkpoint_config_path),
                "source_audit": geometry_settings.get("source_checkpoint_audit", {}),
            },
            "resolved_config": {
                "source_env": env_config,
                "gamma": geometry_settings.get("source_checkpoint_audit", {}).get("gamma"),
                "return_estimator": geometry_settings.get("source_checkpoint_audit", {}).get("return_estimator"),
                "pql_horizon": geometry_settings.get("source_checkpoint_audit", {}).get("pql_horizon"),
                "pql_lambda": geometry_settings.get("source_checkpoint_audit", {}).get("pql_lambda"),
                "policy": checkpoint_config.get("policy"),
                "train": checkpoint_config.get("train"),
            },
            "experiment_env_overrides": {
                "compute_eval_metrics": "disabled for prefix replay, enabled at forced-action boundary",
                "probe_horizon": PROBE_HORIZON,
            },
            "map_hashes": map_hashes,
            "native_build_fingerprint": native_fingerprint,
            "source_hashes": source_file_hashes,
            "run_settings": run_settings,
        })
        run_signature = manifest["signature"]
        _worker_progress(
            records_dir, args.worker_index, args.workers,
            status="running", run_signature=run_signature,
            assigned_state_count=len(assigned_states), completed_state_count=0,
        )

        completed_state_count = 0
        completed_branch_count = 0
        models = None
        try:
            groups = defaultdict(list)
            for state in assigned_states:
                groups[str(state["source_shard"])].append(state)
            for shard_name in sorted(groups):
                states = groups[shard_name]
                shard_path, sidecar_path, episode_metadata, arrays, _, reference = _load_source_episode(
                    source_records, geometry_manifest["signature"], states[0]
                )
                if any(str(state["source_shard"]) != shard_name for state in states):
                    raise RuntimeError("worker shard group contains mixed source paths")
                map_path = Path(episode_metadata["map_path"]).expanduser().resolve()
                map_hash = sha256_file(map_path)
                if map_hash != episode_metadata.get("map_sha256"):
                    raise ValueError(f"source map hash differs from geometry sidecar: {map_path}")
                expected_map_hash = map_hashes.get(str(map_path))
                if expected_map_hash != map_hash:
                    raise ValueError(f"source map is not covered by geometry manifest: {map_path}")
                map_index = int(episode_metadata["map_index"])
                if map_index != int(states[0]["map_index"]):
                    raise ValueError(f"source map index differs at {shard_path}")

                incomplete_states = []
                state_tasks = {}
                for state in states:
                    probe_id = str(state["state_id"])
                    tasks = build_branch_tasks(
                        probe_id, seed, int(state["episode_index"]), int(state["frame_idx"]),
                        int(state["slot"]), repeats=REPEATS, actions=NUM_ACTIONS,
                    )
                    state_tasks[probe_id] = tasks
                    if probe_is_complete(
                        records_dir, probe_id, run_signature,
                        expected_branch_count=NUM_ACTIONS, expected_repeats=REPEATS,
                        expected_action_count=NUM_ACTIONS,
                        expected_task_ids=[task["task_id"] for task in tasks],
                    ):
                        completed_state_count += 1
                        completed_branch_count += NUM_ACTIONS
                        continue
                    incomplete_states.append(state)
                if not incomplete_states:
                    continue

                layout_seed = int(episode_metadata["layout_seed"])
                template = probe_env.create_template(env_config, seed=layout_seed)
                try:
                    agent_count = int(episode_metadata["agent_count"])
                    if template.map_agent_counts.get(str(map_path)) != agent_count:
                        raise ValueError(
                            f"recreated layout agent count differs for {map_path}: "
                            f"expected {agent_count}, got {template.map_agent_counts.get(str(map_path))}"
                        )
                    if models is None:
                        models = FrozenModels(checkpoint_path, template.model_env, device=args.device)
                        expected_hashes = geometry_settings.get("source_checkpoint_audit", {}).get("model_hashes")
                        if expected_hashes and models.initial_hashes != expected_hashes:
                            raise ValueError("loaded frozen model hashes differ from source geometry audit")
                        if models.audit["checkpoint_sha256"] != geometry_settings["source_checkpoint_sha256"]:
                            raise ValueError("loaded checkpoint hash differs from source geometry manifest")
                        if not np.allclose(models.action_table, probe_env._jerk_action_table(), rtol=0.0, atol=1e-7):
                            raise ValueError("checkpoint action table differs from native jerk action order")
                    pql_horizon = int(models.train_config.get("pql_horizon", 128))
                    pql_lambda = float(models.train_config.get("pql_lambda", 0.95))
                    for state in incomplete_states:
                        source_row = _source_row_for_state(state, episode_metadata, arrays)
                        frame_idx = int(state["frame_idx"])
                        ego_idx = int(state["slot"])
                        probe_id = str(state["state_id"])
                        probe = {
                            "frame_idx": frame_idx,
                            "ego_idx": ego_idx,
                            "agent_id": int(state["agent_id"]),
                            "slot": ego_idx,
                            "state_id": probe_id,
                            "assigned_bin": _assigned_bin(state),
                        }
                        def report_branch_progress(frame_count, live_count):
                            if frame_count % 512 == 0:
                                print(json.dumps({
                                    "worker": args.worker_index,
                                    "state_id": probe_id,
                                    "branch_frames": frame_count,
                                    "live_branches": live_count,
                                }, sort_keys=True))

                        task_records = _run_branch_batch(
                            probe_env.SceneBatch, template, str(map_path), agent_count,
                            int(episode_metadata["scene_seed"]), reference, probe,
                            state_tasks[probe_id], models, pql_horizon,
                            progress_callback=report_branch_progress,
                            probe_horizon=PROBE_HORIZON, compact_metadata=True, compact_prefix=True,
                        )
                        _verify_replayed_state(task_records, state, source_row)
                        probe_metadata, output_arrays = _assemble_probe_bundle(
                            probe_id, int(state["episode_index"]), str(map_path), map_hash,
                            int(episode_metadata["scene_seed"]), reference, probe, agent_count,
                            task_records, models, compute_probe_measurements, pql_horizon, pql_lambda,
                            num_goals=int(template.probe_config["num_goals"]),
                        )
                        ego_start = probe_metadata["probe"]["ego_start"]
                        probe_metadata["probe"]["initial_ego_geometry"] = initial_geometry(
                            output_arrays["initial_ego_observation"], template.probe_config,
                            current_goal_idx=int(ego_start["current_goal_idx"]),
                            goal_count=int(ego_start["goal_count"]),
                        )
                        probe_metadata["probe"]["reference_file"] = str(
                            Path(state["source_shard"]).as_posix()
                        )
                        probe_metadata.update({
                            "format_version": FORMAT_VERSION,
                            "state_id": probe_id,
                            "assigned_bin": _assigned_bin(state),
                            "selection_state": state,
                            "source_row": source_row,
                            "source": {
                                "records_dir": str(source_records),
                                "geometry_manifest_signature": geometry_manifest["signature"],
                                "geometry_manifest_path": str(source_records / "geometry_manifest.json"),
                                "shard": str(Path(state["source_shard"]).as_posix()),
                                "shard_sha256": sha256_file(shard_path),
                                "sidecar": str(sidecar_path),
                                "sidecar_sha256": sha256_file(sidecar_path),
                                "source_geometry_row": source_row,
                            },
                            "collection": {
                                "continuation_horizon": PROBE_HORIZON,
                                "prefix_frame_count": frame_idx,
                                "action_count": NUM_ACTIONS,
                                "repeats": REPEATS,
                                "branch_batch_size": NUM_ACTIONS,
                                "continuation_seed_scope": "shared_across_forced_actions_for_each_state",
                                "prefix_replay": "compact_native_joint_action_replay_with_endpoint_fingerprint",
                            },
                            "scene": {
                                **probe_metadata["scene"],
                                "layout_seed": int(episode_metadata["layout_seed"]),
                                "layout_draw_index": int(episode_metadata["layout_draw_index"]),
                                "episode_seed": int(episode_metadata["episode_seed"]),
                                "scene_seed": int(episode_metadata["scene_seed"]),
                                "source_frame_count": int(episode_metadata["frame_count"]),
                                "source_boundary_captured": bool(episode_metadata["boundary_captured"]),
                            },
                            "model_hashes": models.audit["model_hashes"],
                            "run_signature": run_signature,
                            "branch_task_count": NUM_ACTIONS,
                            "repeats": REPEATS,
                            "action_count": NUM_ACTIONS,
                            "branch_task_ids": [task["task_id"] for task in state_tasks[probe_id]],
                        })
                        probe_metadata["scene"]["scene_parameters"] = _scene_parameters(template.probe_config)
                        metadata_path, data_path = probe_paths(records_dir, probe_id)
                        probe_metadata["selection_file"] = str(selection_path.relative_to(records_dir))
                        from .run_probes import save_probe
                        save_probe(records_dir, probe_id, run_signature, probe_metadata, output_arrays)
                        completed_state_count += 1
                        completed_branch_count += NUM_ACTIONS
                        _worker_progress(
                            records_dir, args.worker_index, args.workers,
                            status="running", run_signature=run_signature,
                            assigned_state_count=len(assigned_states),
                            completed_state_count=completed_state_count,
                            completed_branch_count=completed_branch_count,
                            current_state_id=probe_id,
                        )
                        print(json.dumps({
                            "worker": args.worker_index,
                            "state_id": probe_id,
                            "completed_states": completed_state_count,
                            "assigned_states": len(assigned_states),
                            "branches": NUM_ACTIONS,
                            "data_file": data_path.name,
                        }, sort_keys=True))
                        models.verify_frozen()
                finally:
                    template.close()

            summary = _update_collection_summary(records_dir, selection, run_signature)
            _worker_progress(
                records_dir, args.worker_index, args.workers,
                status="complete", run_signature=run_signature,
                assigned_state_count=len(assigned_states),
                completed_state_count=completed_state_count,
                completed_branch_count=completed_branch_count,
                overall_completed_state_count=summary["completed_state_count"],
                overall_expected_state_count=summary["selection_state_count"],
                overall_complete=summary["complete"],
            )
            return summary
        except Exception as error:
            _worker_progress(
                records_dir, args.worker_index, args.workers,
                status="failed", run_signature=run_signature,
                error=f"{type(error).__name__}: {error}",
                completed_state_count=completed_state_count,
                completed_branch_count=completed_branch_count,
            )
            raise
        finally:
            if models is not None:
                models.verify_frozen()


if __name__ == "__main__":
    main()
