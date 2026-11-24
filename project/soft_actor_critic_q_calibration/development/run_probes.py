"""Collect paired frozen-policy first-action probes from held CARLA scenes."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import numpy as np

from .initial_geometry import initial_geometry
from .run_layout import initialize_model_directory, model_directory, records_directory


PROBE_FRACTIONS = (0.10, 0.30, 0.50)
NUM_ACTIONS = 12
DEFAULT_EPISODES = 2
DEFAULT_REPEATS = 8
DEFAULT_BRANCH_BATCH_SIZE = 24
SAFETY_EVENTS = ("collision", "at_fault_collision", "offroad", "red_light")
FORMAT_VERSION = 1
ENV_PROBABILITY_OVERRIDES = {
    "partner_blindness_prob": 0.0,
    "partner_blindness_trigger_prob": 0.0,
    "phantom_braking_prob": 0.0,
    "phantom_braking_trigger_prob": 0.0,
}
EVAL_METRIC_ENV_OVERRIDES = {"compute_eval_metrics": True}
PROBE_ENV_OVERRIDES = {**ENV_PROBABILITY_OVERRIDES, **EVAL_METRIC_ENV_OVERRIDES}


def select_probe_frames(preaction_eligible, fractions=PROBE_FRACTIONS, minimum_eligible_agents=2):
    """Select distinct eligible frame indices nearest the requested span fractions.

    ``preaction_eligible`` is a [time, agent] boolean array captured before each
    action. A frame qualifies only when at least ``minimum_eligible_agents``
    agents are eligible. Selection depends only on those masks and time.
    """
    eligible = np.asarray(preaction_eligible, dtype=np.bool_)
    if eligible.ndim != 2:
        raise ValueError("preaction_eligible must have shape [time, agent]")
    if not fractions or any(not 0.0 <= float(fraction) <= 1.0 for fraction in fractions):
        raise ValueError("fractions must be a non-empty sequence in [0, 1]")
    if minimum_eligible_agents < 1:
        raise ValueError("minimum_eligible_agents must be positive")

    eligible_counts = eligible.sum(axis=1, dtype=np.int64)
    candidate_frames = np.flatnonzero(eligible_counts >= minimum_eligible_agents)
    if candidate_frames.size == 0:
        return {
            "requested_count": len(fractions),
            "selected_count": 0,
            "shortage_count": len(fractions),
            "eligible_span": None,
            "frames": [],
        }

    first_frame = int(candidate_frames[0])
    last_frame = int(candidate_frames[-1])
    target_frames = [first_frame + float(fraction) * (last_frame - first_frame) for fraction in fractions]
    selection_count = min(len(fractions), int(candidate_frames.size))

    # With fewer eligible frames than target fractions, select the target
    # subset that best matches the observed frames. Ties resolve by target
    # order, then earlier frame indices.
    best = None
    for target_indices in itertools.combinations(range(len(fractions)), selection_count):
        assignment = _minimum_cost_ordered_assignment(candidate_frames, target_frames, target_indices)
        if assignment is None:
            continue
        cost, frame_indices = assignment
        candidate = (cost, target_indices, frame_indices)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError("could not assign distinct eligible frames to probe fractions")

    _, target_indices, frame_indices = best
    records = []
    for fraction_idx, frame_idx in zip(target_indices, frame_indices):
        agent_indices = np.flatnonzero(eligible[frame_idx]).astype(np.int64)
        records.append({
            "frame_idx": int(frame_idx),
            "target_fraction": float(fractions[fraction_idx]),
            "eligible_agent_count": int(eligible_counts[frame_idx]),
            "eligible_agent_indices": agent_indices.tolist(),
        })

    return {
        "requested_count": len(fractions),
        "selected_count": len(records),
        "shortage_count": len(fractions) - len(records),
        "eligible_span": {"first_frame": first_frame, "last_frame": last_frame},
        "frames": records,
    }


def _minimum_cost_ordered_assignment(candidate_frames, target_frames, target_indices):
    """Assign distinct increasing candidates to an ordered subset of targets."""
    # Each state stores (absolute frame error, selected frame tuple). At every
    # candidate we either preserve an earlier state or append that candidate.
    state_count = len(target_indices)
    states = [None] * (state_count + 1)
    states[0] = (0.0, ())
    for candidate_frame in candidate_frames:
        candidate_frame = int(candidate_frame)
        for assigned_count in range(state_count, 0, -1):
            previous = states[assigned_count - 1]
            if previous is None:
                continue
            target_frame = target_frames[target_indices[assigned_count - 1]]
            proposal = (previous[0] + abs(candidate_frame - target_frame), previous[1] + (candidate_frame,))
            current = states[assigned_count]
            if current is None or proposal < current:
                states[assigned_count] = proposal
    return states[state_count]


def select_probe_egos(eligible_agent_indices, seed, episode_index, frame_idx, count=2):
    """Choose egos deterministically without reading Q values or outcomes."""
    candidates = np.asarray(eligible_agent_indices, dtype=np.int64)
    if candidates.ndim != 1:
        raise ValueError("eligible_agent_indices must be one-dimensional")
    candidates = np.unique(candidates)
    if count < 1:
        raise ValueError("count must be positive")
    if candidates.size < count:
        return []
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed) & 0xFFFFFFFF,
        int(episode_index) & 0xFFFFFFFF,
        int(frame_idx) & 0xFFFFFFFF,
        0xE601,
    ]))
    selected = rng.choice(candidates, size=count, replace=False)
    return sorted(int(agent_idx) for agent_idx in selected)


def build_probe_selection(preaction_eligible, seed, episode_index, fractions=PROBE_FRACTIONS):
    """Return selected frame/ego pairs and explicit shortage counts."""
    frame_selection = select_probe_frames(preaction_eligible, fractions=fractions)
    probes = []
    shortage_count = frame_selection["shortage_count"] * 2
    for frame in frame_selection["frames"]:
        ego_indices = select_probe_egos(
            frame["eligible_agent_indices"], seed, episode_index, frame["frame_idx"], count=2
        )
        if len(ego_indices) != 2:
            shortage_count += 2 - len(ego_indices)
            continue
        for ego_idx in ego_indices:
            probes.append({
                "frame_idx": frame["frame_idx"],
                "target_fraction": frame["target_fraction"],
                "eligible_agent_count": frame["eligible_agent_count"],
                "ego_idx": ego_idx,
            })
    return {
        "frame_selection": frame_selection,
        "probes": probes,
        "requested_probe_count": len(fractions) * 2,
        "selected_probe_count": len(probes),
        "shortage_count": shortage_count,
    }


def stable_seed(seed, *identity_parts):
    """Derive a stable signed-63-bit seed from task identity, never batch size."""
    identity = json.dumps([int(seed), *identity_parts], ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8, person=b"puffprobe").digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def continuation_seed(seed, episode_index, frame_idx, ego_idx, repeat_idx):
    """Shared seed for all first actions in one paired continuation repeat."""
    return stable_seed(seed, "continuation", episode_index, frame_idx, ego_idx, repeat_idx)


def sample_actions(probabilities, rng):
    """Sample one categorical action per row from an explicit RNG stream."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("probabilities must have shape [batch, actions]")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("policy probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("each policy probability row must sum to one")
    draws = rng.random(probabilities.shape[0])
    cumulative = np.cumsum(probabilities, axis=1)
    # Float32 softmax can leave a small gap below one. Draw in each row's actual mass.
    draws *= cumulative[:, -1]
    action_indices = (draws[:, None] >= cumulative).sum(axis=1)
    if np.any(action_indices >= probabilities.shape[1]):
        raise RuntimeError("categorical sample exceeded the policy action table")
    return action_indices.astype(np.int32, copy=False)


def build_branch_tasks(probe_id, seed, episode_index, frame_idx, ego_idx, repeats=DEFAULT_REPEATS,
                       actions=NUM_ACTIONS):
    """Create action/repeat tasks with action-paired, batch-independent seeds."""
    if repeats <= 0 or actions <= 0:
        raise ValueError("repeats and actions must be positive")
    tasks = []
    for repeat_idx in range(repeats):
        paired_seed = continuation_seed(seed, episode_index, frame_idx, ego_idx, repeat_idx)
        for action_idx in range(actions):
            tasks.append({
                "task_id": f"{probe_id}/repeat_{repeat_idx:03d}/action_{action_idx:02d}",
                "probe_id": str(probe_id),
                "repeat_idx": repeat_idx,
                "action_idx": action_idx,
                "continuation_seed": paired_seed,
            })
    return tasks


def chunked(items: Iterable, chunk_size: int):
    """Yield stable chunks; chunking does not alter task identities or seeds."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def sanitize_json(value):
    """Convert arrays/scalars and represent every non-finite number as null."""
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_payload = sanitize_json(payload)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        json.dump(clean_payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def atomic_write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def current_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_hashes(paths):
    hashes = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_signature(manifest):
    immutable_fields = {
        key: manifest[key]
        for key in (
            "format_version", "checkpoint", "resolved_config", "experiment_env_overrides",
            "map_hashes", "native_build_fingerprint", "source_hashes", "run_settings",
        )
        if key in manifest
    }
    encoded = json.dumps(sanitize_json(immutable_fields), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def open_or_create_manifest(output_dir, manifest):
    """Resume only when all immutable checkpoint/config/source/run inputs match."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    proposed = dict(manifest)
    proposed["format_version"] = FORMAT_VERSION
    proposed["created_at"] = current_utc()
    proposed["signature"] = manifest_signature(proposed)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("signature") != proposed["signature"]:
            raise ValueError("output manifest does not match checkpoint, config, source, or run settings")
        return existing
    atomic_write_json(path, proposed)
    return proposed


def probe_paths(output_dir, probe_id):
    safe_id = str(probe_id).replace("/", "__")
    probe_dir = Path(output_dir) / "probes"
    return probe_dir / f"{safe_id}.json", probe_dir / f"{safe_id}.npz"


def probe_is_complete(output_dir, probe_id, run_signature, expected_branch_count=None,
                      expected_repeats=None, expected_action_count=None, expected_task_ids=None):
    metadata_path, data_path = probe_paths(output_dir, probe_id)
    if not metadata_path.is_file() or not data_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("status") != "complete" or metadata.get("run_signature") != run_signature:
        return False
    if metadata.get("probe_id") != str(probe_id):
        return False
    if metadata.get("data_sha256") != sha256_file(data_path):
        return False
    for field, expected in (
        ("branch_task_count", expected_branch_count),
        ("repeats", expected_repeats),
        ("action_count", expected_action_count),
    ):
        if expected is not None and metadata.get(field) != expected:
            return False
    if expected_task_ids is not None and metadata.get("branch_task_ids") != list(expected_task_ids):
        return False
    return True


def save_probe(output_dir, probe_id, run_signature, metadata, arrays):
    metadata_path, data_path = probe_paths(output_dir, probe_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(data_path, **arrays)
    complete_metadata = dict(metadata)
    complete_metadata.update({
        "status": "complete",
        "probe_id": str(probe_id),
        "run_signature": run_signature,
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "completed_at": current_utc(),
    })
    atomic_write_json(metadata_path, complete_metadata)


def update_progress(output_dir, **fields):
    path = Path(output_dir) / "progress.json"
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(fields)
    current["updated_at"] = current_utc()
    atomic_write_json(path, current)


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--branch-batch-size", type=int, default=DEFAULT_BRANCH_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--compile-actor", action="store_true")
    parser.add_argument("--reference-cap", type=int, default=None,
                        help="execution gate: cap captured reference frames per scene episode")
    parser.add_argument("--max-probes", type=int, default=None,
                        help="execution gate: stop after this many selected ego probes")
    return parser


def validate_args(args):
    for name in ("episodes", "repeats", "branch_batch_size", "cpu_threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.reference_cap is not None and args.reference_cap <= 0:
        raise ValueError("reference-cap must be positive")
    if args.max_probes is not None and args.max_probes <= 0:
        raise ValueError("max-probes must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        run_experiment(args)
    except Exception as error:
        try:
            update_progress(
                records_directory(initialize_model_directory(args.output)), status="failed",
                error=f"{type(error).__name__}: {error}",
            )
        except Exception:
            # Keep the original collection error when the supplied model path
            # is invalid or its progress file cannot be written.
            pass
        raise


def run_experiment(args):
    """Run reference collection and paired branches with the private native API."""
    import torch
    import yaml

    development_dir = Path(__file__).resolve().parent
    repo_root = development_dir.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    package_name = "project.soft_actor_critic_q_calibration"
    development_package = f"{package_name}.development"
    probe_env = importlib.import_module(f"{development_package}.probe_env")
    model_module = importlib.import_module(f"{development_package}.models")
    returns_module = importlib.import_module(f"{development_package}.probe_returns")
    FrozenModels = model_module.FrozenModels
    compute_probe_measurements = returns_module.compute_probe_measurements

    torch.set_num_threads(args.cpu_threads)
    model_dir = initialize_model_directory(model_directory(args.output))
    output_dir = records_directory(model_dir)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_config_path = checkpoint_path.with_name("config.yaml")
    if not checkpoint_config_path.is_file():
        raise FileNotFoundError(f"checkpoint config not found: {checkpoint_config_path}")
    checkpoint_config = yaml.safe_load(checkpoint_config_path.read_text(encoding="utf-8"))
    if not isinstance(checkpoint_config, dict) or not isinstance(checkpoint_config.get("env"), dict):
        raise ValueError(f"checkpoint config must contain an env mapping: {checkpoint_config_path}")
    saved_env_config = dict(checkpoint_config["env"])
    env_config = dict(saved_env_config)
    map_dir = Path(env_config["map_dir"]).expanduser()
    if not map_dir.is_absolute():
        map_dir = repo_root / map_dir
    env_config["map_dir"] = str(map_dir.resolve())
    env_config.update(PROBE_ENV_OVERRIDES)

    output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(
        output_dir, status="initializing", checkpoint=str(checkpoint_path), device=args.device,
        requested_episodes=args.episodes, requested_repeats=args.repeats,
        branch_batch_size=args.branch_batch_size, completed_episodes=0,
        completed_probes=0, completed_branches=0,
    )
    template = None
    try:
        template = probe_env.create_template(env_config, seed=int(args.seed))
        if template.template_agent_count <= 0:
            raise RuntimeError("native template selected no agent slots")
        map_paths = [str(Path(path).resolve()) for path in template.map_paths]
        if not map_paths:
            raise RuntimeError("native template exposed no training maps")
        episode_map_paths = [
            map_paths[stable_seed(args.seed, "map", episode_idx) % len(map_paths)]
            for episode_idx in range(args.episodes)
        ]
        map_agent_counts = {
            str(Path(path).resolve()): int(count)
            for path, count in template.map_agent_counts.items()
        }
        episode_agent_counts = [map_agent_counts[path] for path in episode_map_paths]
        unique_maps = sorted(set(episode_map_paths))
        map_hashes = {path: sha256_file(path) for path in unique_maps}

        private_binding = probe_env.ensure_private_binding()
        native_build_fingerprint = _native_build_fingerprint(
            private_binding, probe_env.native_build_fingerprint()
        )
        source_paths = [
            Path(__file__),
            development_dir / "models.py",
            development_dir / "probe_returns.py",
            development_dir / "initial_geometry.py",
            development_dir / "probe_env.py",
            development_dir / "native_probe" / "build_private_binding.py",
            development_dir / "native_probe" / "probe_export.h",
            repo_root / "pufferlib" / "sac_returns.py",
            repo_root / "pufferlib" / "ocean" / "drive" / "drive.h",
            repo_root / "pufferlib" / "ocean" / "drive" / "binding.c",
            repo_root / "pufferlib" / "ocean" / "drive" / "constants.h",
            repo_root / "pufferlib" / "ocean" / "env_binding.h",
        ]
        hashes = source_hashes(source_paths)
        hashes.update(source_hashes([
            repo_root / "pufferlib" / "ocean" / "torch_sac.py",
            repo_root / "pufferlib" / "ocean" / "torch.py",
            repo_root / "pufferlib" / "ocean" / "drive" / "drive.py",
        ]))
        models = FrozenModels(
            checkpoint_path, template.model_env, device=args.device, compile_actor=args.compile_actor
        )
        models.verify_frozen()
        pql_horizon = int(models.train_config.get("pql_horizon", 128))
        pql_lambda = float(models.train_config.get("pql_lambda", 0.95))
        scenario_length = env_config.get("scenario_length")
        if scenario_length is None or int(scenario_length) <= 0:
            raise ValueError("saved environment must have a positive scenario_length")
        max_reference_frames = args.reference_cap or int(scenario_length)
        run_settings = {
            "episodes": int(args.episodes),
            "probe_fractions": list(PROBE_FRACTIONS),
            "agents_per_probe_frame": 2,
            "repeats": int(args.repeats),
            "action_count": NUM_ACTIONS,
            "branch_batch_size": int(args.branch_batch_size),
            "seed": int(args.seed),
            "device": str(args.device),
            "cpu_threads": int(args.cpu_threads),
            "compile_actor": bool(args.compile_actor),
            "reference_cap": args.reference_cap,
            "max_probes": args.max_probes,
            "episode_maps": episode_map_paths,
            "episode_agent_counts": episode_agent_counts,
            "action_table_normalized": models.action_table.tolist(),
            "continuation_rng_scope": "policy_action_sampling_only",
            "eval_metric_collection": {
                "enabled": True,
                "env_override": EVAL_METRIC_ENV_OVERRIDES,
                "scope": "probing_agent_forced_action_to_branch_end",
                "sample_count_per_branch": 1,
                "safety_event_flags_order": list(SAFETY_EVENTS),
                "completion_boundary": "terminal_or_truncated_or_validity_gap",
            },
        }
        manifest = open_or_create_manifest(output_dir, {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "config_path": str(checkpoint_config_path),
                "config_sha256": sha256_file(checkpoint_config_path),
                "audit": models.audit,
            },
            "resolved_config": {
                "saved_env": saved_env_config,
                "resolved_env": template.probe_config,
                "gamma": models.gamma,
                "alpha": models.alpha,
                "return_estimator": models.return_estimator,
                "pql_horizon": pql_horizon,
                "pql_lambda": pql_lambda,
            },
            "experiment_env_overrides": PROBE_ENV_OVERRIDES,
            "map_hashes": map_hashes,
            "native_build_fingerprint": native_build_fingerprint,
            "source_hashes": hashes,
            "run_settings": run_settings,
        })
        run_signature = manifest["signature"]
        update_progress(
            output_dir, status="running", run_signature=run_signature,
            selected_map_count=len(unique_maps), native_build_fingerprint=native_build_fingerprint,
            model_audit=models.audit, pql_horizon=pql_horizon,
        )

        completed_probes = 0
        completed_branches = 0
        selected_probe_count = 0
        for episode_idx, map_path in enumerate(episode_map_paths):
            if args.max_probes is not None and selected_probe_count >= args.max_probes:
                break
            map_sha = map_hashes[map_path]
            agent_count = episode_agent_counts[episode_idx]
            scene_seed = stable_seed(args.seed, "scene", episode_idx)
            reference = _load_reference(
                output_dir, episode_idx, run_signature, map_path, map_sha
            )
            if reference is None:
                reference = _collect_reference(
                    probe_env.SceneBatch, template, map_path, agent_count,
                    scene_seed, episode_idx, args, models, max_reference_frames,
                    run_signature, map_sha,
                    progress_callback=lambda frame_count, boundary: update_progress(
                        output_dir, status="running", phase="reference",
                        current_episode=episode_idx, reference_frames_processed=frame_count,
                        reference_boundary_captured=boundary,
                    ) if frame_count % 64 == 0 or boundary else None,
                )
                _save_reference(output_dir, episode_idx, run_signature, reference)
            update_progress(
                output_dir, status="running", completed_episodes=episode_idx + 1,
                reference_frames=reference["frame_count"],
                reference_boundary_captured=reference["boundary_captured"],
                completed_probes=completed_probes, completed_branches=completed_branches,
            )
            selection = build_probe_selection(reference["preaction_eligible"], args.seed, episode_idx)
            probes = selection["probes"]
            natural_probe_count = len(probes)
            if args.max_probes is not None:
                remaining = max(args.max_probes - selected_probe_count, 0)
                probes = probes[:remaining]
            capped_probe_count = len(probes)
            selection_path = Path(output_dir) / "references" / f"episode_{episode_idx:03d}_selection.json"
            selection_record = {
                "episode_index": episode_idx,
                "frame_selection": selection["frame_selection"],
                "requested_probe_count": selection["requested_probe_count"],
                "naturally_selected_probe_count": natural_probe_count,
                "selected_probe_count": capped_probe_count,
                "eligibility_shortage_count": selection["shortage_count"],
                "execution_gate_probe_cap": args.max_probes,
                "selected_probes": probes,
            }
            atomic_write_json(selection_path, selection_record)
            update_progress(
                output_dir, status="running", phase="selection", current_episode=episode_idx,
                frame_shortage_count=selection["frame_selection"]["shortage_count"],
                eligibility_probe_shortage_count=selection["shortage_count"],
                selected_probes_this_episode=capped_probe_count,
                selection_file=str(selection_path.relative_to(output_dir)),
            )
            selected_probe_count += len(probes)
            if not probes:
                update_progress(
                    output_dir, last_episode_shortage=selection["shortage_count"],
                    completed_probes=completed_probes, completed_branches=completed_branches,
                )
                continue

            for probe in probes:
                probe_id = _probe_id(episode_idx, probe["frame_idx"], probe["ego_idx"])
                tasks = build_branch_tasks(
                    probe_id, args.seed, episode_idx, probe["frame_idx"], probe["ego_idx"],
                    repeats=args.repeats, actions=NUM_ACTIONS,
                )
                if probe_is_complete(
                    output_dir, probe_id, run_signature, expected_branch_count=len(tasks),
                    expected_repeats=args.repeats, expected_action_count=NUM_ACTIONS,
                    expected_task_ids=[task["task_id"] for task in tasks],
                ):
                    completed_probes += 1
                    completed_branches += len(tasks)
                    update_progress(
                        output_dir, status="running", completed_probes=completed_probes,
                        completed_branches=completed_branches, last_probe_id=probe_id,
                    )
                    continue

                task_records = [None] * len(tasks)
                for start_idx, task_batch in _indexed_chunks(tasks, args.branch_batch_size):
                    branch_results = _run_branch_batch(
                        probe_env.SceneBatch, template, map_path, agent_count,
                        scene_seed, reference, probe, task_batch, models, pql_horizon,
                        progress_callback=lambda frame_count, live_count, batch_start=start_idx, batch_size=len(task_batch): update_progress(
                            output_dir, status="running", phase="branching",
                            current_episode=episode_idx, current_probe_id=probe_id,
                            branch_batches_finished=batch_start // args.branch_batch_size,
                            current_probe_branches_finished=batch_start,
                            active_branch_batch_size=batch_size,
                            current_probe_branch_count=len(tasks),
                            branch_frames_processed=frame_count, live_branches=live_count,
                        ) if frame_count % 64 == 0 else None,
                    )
                    for offset, result in enumerate(branch_results):
                        task_records[start_idx + offset] = result
                    update_progress(
                        output_dir, status="running", phase="branching",
                        current_episode=episode_idx, current_probe_id=probe_id,
                        branch_batches_finished=(start_idx + len(task_batch) + args.branch_batch_size - 1)
                        // args.branch_batch_size,
                        current_probe_branches_finished=start_idx + len(task_batch),
                        current_probe_branch_count=len(tasks),
                    )

                probe_metadata, arrays = _assemble_probe_bundle(
                    probe_id, episode_idx, map_path, map_sha, scene_seed, reference, probe,
                    agent_count, task_records, models, compute_probe_measurements, pql_horizon, pql_lambda,
                    num_goals=int(template.probe_config["num_goals"]),
                )
                probe_metadata.update({
                    "run_signature": run_signature,
                    "branch_task_count": len(tasks),
                    "repeats": int(args.repeats),
                    "action_count": NUM_ACTIONS,
                    "branch_task_ids": [task["task_id"] for task in tasks],
                    "reference_boundary_captured": bool(reference["boundary_captured"]),
                    "reference_cap_reached": bool(reference["reference_cap_reached"]),
                })
                ego_start = probe_metadata["probe"]["ego_start"]
                probe_metadata["probe"]["initial_ego_geometry"] = initial_geometry(
                    arrays["initial_ego_observation"], template.probe_config,
                    current_goal_idx=int(ego_start["current_goal_idx"]),
                    goal_count=int(ego_start["goal_count"]),
                )
                probe_metadata["selection"] = {
                    **selection_record,
                    "selection_file": str(selection_path.relative_to(output_dir)),
                }
                probe_metadata["scene"]["scene_parameters"] = _scene_parameters(template.probe_config)
                save_probe(output_dir, probe_id, run_signature, probe_metadata, arrays)
                completed_probes += 1
                completed_branches += len(tasks)
                update_progress(
                    output_dir, status="running", completed_probes=completed_probes,
                    completed_branches=completed_branches, last_probe_id=probe_id,
                    last_completed_at=current_utc(),
                )
                models.verify_frozen()

        models.verify_frozen()
        update_progress(
            output_dir, status="complete", completed_episodes=min(args.episodes, episode_idx + 1),
            completed_probes=completed_probes, completed_branches=completed_branches,
            selected_probe_count=selected_probe_count, run_signature=run_signature,
            execution_gate=(args.reference_cap is not None or args.max_probes is not None or args.repeats != DEFAULT_REPEATS),
        )
    finally:
        if template is not None:
            template.close()


def _native_build_fingerprint(private_binding, source_fingerprint):
    path = Path(private_binding.__file__).resolve()
    return {
        "private_binding_path": str(path),
        "private_binding_sha256": sha256_file(path),
        "source_fingerprint": str(source_fingerprint),
    }


def _scene_parameters(config):
    names = (
        "dt", "scenario_length", "termination_mode", "inactive_agent_threshold", "control_mode",
        "simulation_mode", "dynamics_model", "collision_behavior", "offroad_behavior",
        "traffic_light_behavior", "goal_regen_mode", "goal_source", "goal_radius", "num_goals",
        "reward_conditioning", "reward_randomization", "reward_collision", "reward_offroad",
        "reward_goal", "reward_stop_line", "partner_blindness_prob", "phantom_braking_prob",
    )
    return {name: config[name] for name in names if name in config}


def _probe_id(episode_idx, frame_idx, ego_idx):
    return f"episode_{episode_idx:03d}/frame_{frame_idx:05d}/ego_{ego_idx:04d}"


def _indexed_chunks(items, chunk_size):
    for start_idx in range(0, len(items), chunk_size):
        yield start_idx, items[start_idx:start_idx + chunk_size]


def _metadata_copy(metadata, copy_idx=0):
    result = {}
    for name, value in metadata.items():
        array = np.asarray(value)
        if name in {"timestep", "episode_seed", "scene_done"}:
            result[name] = np.asarray(array[copy_idx]).copy()
        elif array.ndim >= 1:
            result[name] = np.asarray(array[copy_idx]).copy()
    return result


def _scene_done(metadata, copy_idx):
    values = np.asarray(metadata["scene_done"], dtype=np.bool_).reshape(-1)
    return bool(values[copy_idx])


def observation_fingerprint(observations):
    array = np.ascontiguousarray(observations, dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _collect_reference(
    scene_batch_type, template, map_path, agent_count, scene_seed, episode_idx,
    args, models, max_reference_frames, run_signature, map_sha, progress_callback=None,
):
    actions = []
    eligibility = []
    fingerprints = []
    observation_fingerprints = []
    metadata_series = {}
    reference_rng = np.random.default_rng(stable_seed(args.seed, "reference-policy", episode_idx))
    batch = scene_batch_type.from_template(template, map_path, agent_count, 1, scene_seed)
    boundary_captured = False
    try:
        batch.reset()
        for frame_idx in range(max_reference_frames):
            metadata = batch.metadata()
            eligible = np.asarray(metadata["eligible"], dtype=np.bool_)
            if eligible.shape != (1, agent_count):
                raise RuntimeError(f"unexpected reference eligibility shape: {eligible.shape}")
            fingerprints.append(batch.fingerprint(0))
            eligibility.append(eligible[0].copy())
            for name, row in _metadata_copy(metadata).items():
                metadata_series.setdefault(name, []).append(row)

            observations = np.asarray(batch.observations, dtype=np.float32)
            if observations.shape[0] != 1 or observations.shape[1] != agent_count:
                raise RuntimeError(f"unexpected reference observation shape: {observations.shape}")
            observation_fingerprints.append(observation_fingerprint(observations[0]))
            probabilities, _ = models.policy(observations[0])
            if probabilities.shape[0] != agent_count:
                raise RuntimeError("policy output agent count did not match native scene capacity")
            joint_action = sample_actions(probabilities, reference_rng)
            actions.append(joint_action.copy())
            step_result = batch.step(models.action_table[joint_action][None, ...])
            if _scene_done(step_result["metadata"], 0):
                boundary_captured = True
                if progress_callback is not None:
                    progress_callback(frame_idx + 1, boundary_captured)
                break
            if progress_callback is not None:
                progress_callback(frame_idx + 1, boundary_captured)
        reference_cap_reached = not boundary_captured and args.reference_cap is not None
        if not boundary_captured and not reference_cap_reached:
            raise RuntimeError(
                f"native reference did not hold at its scene boundary within {max_reference_frames} frames"
            )
    finally:
        batch.close()

    if not actions:
        raise RuntimeError(f"reference episode {episode_idx} contained no action frames")
    arrays = {
        "joint_actions": np.stack(actions).astype(np.int32, copy=False),
        "preaction_eligible": np.stack(eligibility).astype(np.bool_, copy=False),
        "fingerprints": np.asarray(fingerprints, dtype="U64"),
        "observation_fingerprints": np.asarray(observation_fingerprints, dtype="U64"),
    }
    for name, rows in metadata_series.items():
        arrays[f"metadata__{name}"] = np.stack(rows)
    first_metadata = {name: rows[0] for name, rows in metadata_series.items()}
    return {
        "episode_index": int(episode_idx),
        "map_path": str(map_path),
        "map_sha256": map_sha,
        "scene_seed": int(scene_seed),
        "run_signature": run_signature,
        "frame_count": len(actions),
        "boundary_captured": boundary_captured,
        "reference_cap_reached": reference_cap_reached,
        "arrays": arrays,
        "preaction_eligible": arrays["preaction_eligible"],
        "joint_actions": arrays["joint_actions"],
        "fingerprints": arrays["fingerprints"].tolist(),
        "observation_fingerprints": arrays["observation_fingerprints"].tolist(),
        "metadata_series": {name: np.stack(rows) for name, rows in metadata_series.items()},
        "initial_metadata": first_metadata,
    }


def _reference_paths(output_dir, episode_idx):
    directory = Path(output_dir) / "references"
    return directory / f"episode_{episode_idx:03d}.json", directory / f"episode_{episode_idx:03d}.npz"


def _save_reference(output_dir, episode_idx, run_signature, reference):
    metadata_path, data_path = _reference_paths(output_dir, episode_idx)
    atomic_write_npz(data_path, **reference["arrays"])
    metadata = {
        "episode_index": episode_idx,
        "run_signature": run_signature,
        "map_path": reference["map_path"],
        "map_sha256": reference["map_sha256"],
        "scene_seed": reference["scene_seed"],
        "frame_count": reference["frame_count"],
        "boundary_captured": reference["boundary_captured"],
        "reference_cap_reached": reference["reference_cap_reached"],
        "initial_metadata_summary": {
            key: value for key, value in reference["initial_metadata"].items()
            if key in {"timestep", "episode_seed", "scene_done"}
        },
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "completed_at": current_utc(),
    }
    atomic_write_json(metadata_path, metadata)


def _load_reference(output_dir, episode_idx, run_signature, map_path, map_sha):
    metadata_path, data_path = _reference_paths(output_dir, episode_idx)
    if not metadata_path.is_file() or not data_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        metadata.get("run_signature") != run_signature
        or metadata.get("map_path") != map_path
        or metadata.get("map_sha256") != map_sha
        or metadata.get("data_sha256") != sha256_file(data_path)
    ):
        return None
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    metadata_series = {
        name[len("metadata__"):]: value
        for name, value in arrays.items() if name.startswith("metadata__")
    }
    return {
        **metadata,
        "arrays": arrays,
        "preaction_eligible": arrays["preaction_eligible"],
        "joint_actions": arrays["joint_actions"],
        "fingerprints": arrays["fingerprints"].astype(str).tolist(),
        "observation_fingerprints": arrays["observation_fingerprints"].astype(str).tolist(),
        "metadata_series": metadata_series,
        "initial_metadata": {name: values[0] for name, values in metadata_series.items()},
    }


def _run_branch_batch(
    scene_batch_type, template, map_path, agent_count, scene_seed, reference,
    probe, tasks, models, pql_horizon, progress_callback=None, *,
    probe_horizon=None, compact_metadata=False, compact_prefix=False,
    record_horizons=(),
):
    copy_count = len(tasks)
    frame_idx = int(probe["frame_idx"])
    ego_idx = int(probe["ego_idx"])
    batch = scene_batch_type.from_template(template, map_path, agent_count, copy_count, scene_seed)
    rngs = None
    branch_records = [
        {
            "task": task,
            "rewards": [], "log_probs": [], "valid": [], "terminals": [],
            "truncations": [], "next_soft_values": [],
            "events": {name: [] for name in SAFETY_EVENTS},
            "goals": [], "validity_gap": False, "scene_boundary": False,
            "initial_q1": None, "initial_q2": None,
            "metrics_mean": None,
            "horizon_metrics": {},
            "metrics_agents_per_batch": copy_count * int(agent_count),
        }
        for task in tasks
    ]
    try:
        batch.reset()
        native_action_table = np.asarray(batch.action_table, dtype=np.float32)
        if native_action_table.shape != models.action_table.shape or not np.allclose(
            native_action_table, models.action_table, rtol=0.0, atol=1e-6
        ):
            raise AssertionError("checkpoint and native discrete action tables do not match")
        if compact_prefix:
            batch.replay_prefix(
                reference["joint_actions"][:frame_idx], frame_idx,
                expected_fingerprint=reference["fingerprints"][frame_idx], compact=True,
            )
        else:
            batch.replay_prefix(
                reference["joint_actions"][:frame_idx], frame_idx,
                expected_fingerprint=reference["fingerprints"][frame_idx],
            )
        fingerprints = [batch.fingerprint(copy_idx) for copy_idx in range(copy_count)]
        if any(fingerprint != reference["fingerprints"][frame_idx] for fingerprint in fingerprints):
            raise AssertionError("branch clone failed the reference fingerprint check")
        rngs = batch.set_continuation_seeds([task["continuation_seed"] for task in tasks])
        observations = np.asarray(batch.observations, dtype=np.float32)
        if observations.ndim != 3 or observations.shape[:2] != (copy_count, agent_count):
            raise RuntimeError(f"unexpected branch observation shape: {observations.shape}")
        expected_observation_fingerprint = reference["observation_fingerprints"][frame_idx]
        if any(
            observation_fingerprint(observations[copy_idx]) != expected_observation_fingerprint
            for copy_idx in range(copy_count)
        ):
            raise AssertionError("replayed branch observation does not match the reference observation")
        if probe_horizon is not None:
            replay_start_metadata = _metadata_copy(batch.metadata(), 0)
            replay_start_fingerprint = batch.fingerprint(0)
            replay_start_observation_fingerprint = observation_fingerprint(observations[0])
            batch.start_probe_horizon(int(probe_horizon))
            for record in branch_records:
                record["replay_start_metadata"] = replay_start_metadata
                record["replay_start_fingerprint"] = replay_start_fingerprint
                record["replay_start_observation_fingerprint"] = replay_start_observation_fingerprint
        # The native buffer is overwritten by the first branch step. Every copy
        # above was checked against the same pre-action observation fingerprint.
        branch_records[0]["initial_ego_observation"] = observations[0, ego_idx].copy()
        initial_q1, initial_q2 = models.q_values(observations[:, ego_idx])
        for copy_idx, record in enumerate(branch_records):
            forced_action = int(tasks[copy_idx]["action_idx"])
            record["initial_q1"] = float(initial_q1[copy_idx, forced_action])
            record["initial_q2"] = float(initial_q2[copy_idx, forced_action])

        metrics_baseline = batch.begin_metrics(ego_idx)
        metric_event_flags = np.zeros((copy_count, len(SAFETY_EVENTS)), dtype=np.bool_)

        live = np.ones(copy_count, dtype=np.bool_)
        scenario_limit = int(
            template.probe_config["scenario_length"] if probe_horizon is None else probe_horizon
        )
        if scenario_limit <= 0:
            raise ValueError(f"branch horizon must be positive, got {scenario_limit}")
        for time_idx in range(scenario_limit):
            if not np.any(live):
                break
            current_observations = np.asarray(batch.observations, dtype=np.float32)
            flat_observations = current_observations.reshape(copy_count * agent_count, -1)
            probabilities_flat, log_probs_flat = models.policy(flat_observations)
            probabilities = probabilities_flat.reshape(copy_count, agent_count, NUM_ACTIONS)
            log_probs = log_probs_flat.reshape(copy_count, agent_count, NUM_ACTIONS)
            discrete_actions = np.zeros((copy_count, agent_count), dtype=np.int32)
            forced_action_ids = np.asarray([task["action_idx"] for task in tasks], dtype=np.int32)
            for copy_idx in np.flatnonzero(live):
                discrete_actions[copy_idx] = sample_actions(probabilities[copy_idx], rngs[copy_idx])
            if time_idx == 0:
                discrete_actions[np.arange(copy_count), ego_idx] = forced_action_ids
            policy_log_probs = log_probs[np.arange(copy_count), ego_idx, discrete_actions[:, ego_idx]]
            normalized_actions = models.action_table[discrete_actions]
            if compact_metadata:
                step_result = batch.step(normalized_actions, compact_metadata=True)
            else:
                step_result = batch.step(normalized_actions)
            next_observations = np.asarray(step_result["observations"], dtype=np.float32)
            next_metadata = step_result["metadata"]
            rewards = np.asarray(step_result["rewards"], dtype=np.float32).reshape(copy_count, agent_count)
            terminals = np.asarray(step_result["terminals"], dtype=np.bool_).reshape(copy_count, agent_count)
            truncations = np.asarray(step_result["truncations"], dtype=np.bool_).reshape(copy_count, agent_count)
            events = step_result["events"]
            for event_name in (*SAFETY_EVENTS, "reached_goal"):
                if event_name not in events:
                    raise RuntimeError(f"native branch step omitted event flag {event_name!r}")
                events[event_name] = np.asarray(events[event_name], dtype=np.bool_).reshape(copy_count, agent_count)
            post_eligible = np.asarray(next_metadata["eligible"], dtype=np.bool_).reshape(copy_count, agent_count)
            scene_done = np.asarray(next_metadata["scene_done"], dtype=np.bool_).reshape(copy_count)

            bootstrap_copy_indices = []
            bootstrap_observations = []
            completed_copy_indices = []
            for copy_idx in np.flatnonzero(live):
                record = branch_records[copy_idx]
                terminal = bool(terminals[copy_idx, ego_idx])
                truncated = bool(truncations[copy_idx, ego_idx])
                boundary = bool(scene_done[copy_idx])
                if boundary and not terminal and not truncated:
                    raise RuntimeError(
                        "native scene boundary failed to mark the probe ego terminal or truncated"
                    )
                validity_gap = not post_eligible[copy_idx, ego_idx] and not terminal and not truncated
                record["rewards"].append(float(rewards[copy_idx, ego_idx]))
                record["log_probs"].append(float(policy_log_probs[copy_idx]))
                record["valid"].append(True)
                record["terminals"].append(terminal)
                record["truncations"].append(truncated)
                record["validity_gap"] = record["validity_gap"] or bool(validity_gap)
                record["scene_boundary"] = record["scene_boundary"] or boundary
                if probe_horizon is not None and time_idx + 1 == scenario_limit:
                    record["probe_horizon_reached"] = bool(truncated)
                for event_position, event_name in enumerate(SAFETY_EVENTS):
                    event_value = bool(events[event_name][copy_idx, ego_idx])
                    record["events"][event_name].append(event_value)
                    metric_event_flags[copy_idx, event_position] |= event_value
                record["goals"].append(bool(events["reached_goal"][copy_idx, ego_idx]))
                done = terminal or truncated or validity_gap
                record["next_soft_values"].append(np.nan)
                if not terminal and not validity_gap and (
                    time_idx + 1 <= pql_horizon or done
                ):
                    bootstrap_copy_indices.append(copy_idx)
                    bootstrap_observations.append(next_observations[copy_idx, ego_idx])
                if done:
                    live[copy_idx] = False
                    completed_copy_indices.append(int(copy_idx))

            horizon = time_idx + 1
            capture_horizon = horizon in record_horizons
            if completed_copy_indices or capture_horizon:
                metrics_by_copy = batch.metrics(metrics_baseline, metric_event_flags)
                if not isinstance(metrics_by_copy, (list, tuple)) or len(metrics_by_copy) != copy_count:
                    raise RuntimeError(
                        f"native metrics returned {len(metrics_by_copy) if hasattr(metrics_by_copy, '__len__') else 'no'} "
                        f"records for {copy_count} branch copies"
                    )
                for copy_idx in completed_copy_indices:
                    branch_records[copy_idx]["metrics_mean"] = _normalize_eval_metrics(
                        metrics_by_copy[copy_idx]
                    )
                if capture_horizon:
                    for copy_idx, record in enumerate(branch_records):
                        if len(record["valid"]) != horizon:
                            continue
                        rewards_to_horizon = np.asarray(record["rewards"], dtype=np.float64)
                        log_probs_to_horizon = np.asarray(record["log_probs"], dtype=np.float64)
                        discounts = np.power(models.gamma, np.arange(horizon, dtype=np.float64))
                        discounted_return = float(np.dot(discounts, rewards_to_horizon))
                        entropy_penalty = float(
                            np.dot(discounts[1:], models.alpha * log_probs_to_horizon[1:])
                        )
                        record["horizon_metrics"][str(horizon)] = {
                            "metrics_mean": _normalize_eval_metrics(metrics_by_copy[copy_idx]),
                            "discounted_reward_return": discounted_return,
                            "standard_soft_return": discounted_return - entropy_penalty,
                        }

            if bootstrap_copy_indices:
                target_values = models.next_values(np.asarray(bootstrap_observations, dtype=np.float32))
                for copy_idx, target_value in zip(bootstrap_copy_indices, target_values):
                    branch_records[copy_idx]["next_soft_values"][-1] = float(target_value)
            if np.any(scene_done & live):
                # A scene boundary must have ended every still-measured ego.
                raise RuntimeError("native scene boundary did not emit a truncation for a live probe ego")
            if progress_callback is not None:
                progress_callback(time_idx + 1, int(live.sum()))
        if np.any(live):
            raise RuntimeError(f"branch continuation exceeded configured horizon={scenario_limit}")
        return branch_records
    finally:
        batch.close()


def _assemble_probe_bundle(
    probe_id, episode_idx, map_path, map_sha, scene_seed, reference, probe,
    agent_count, branch_records, models, compute_probe_measurements, pql_horizon, pql_lambda,
    *, num_goals, record_horizons=(),
):
    branch_count = len(branch_records)
    observed_horizon = max(len(record["valid"]) for record in branch_records)
    shape = (observed_horizon, branch_count)
    rewards = np.zeros(shape, dtype=np.float64)
    log_probs = np.zeros(shape, dtype=np.float64)
    valid = np.zeros(shape, dtype=np.bool_)
    terminals = np.zeros(shape, dtype=np.bool_)
    truncations = np.zeros(shape, dtype=np.bool_)
    next_soft_values = np.full(shape, np.nan, dtype=np.float64)
    safety_arrays = {name: np.zeros(shape, dtype=np.bool_) for name in SAFETY_EVENTS}
    goal_events = np.zeros(shape, dtype=np.bool_)
    for branch_idx, record in enumerate(branch_records):
        count = len(record["valid"])
        rewards[:count, branch_idx] = record["rewards"]
        log_probs[:count, branch_idx] = record["log_probs"]
        valid[:count, branch_idx] = record["valid"]
        terminals[:count, branch_idx] = record["terminals"]
        truncations[:count, branch_idx] = record["truncations"]
        next_soft_values[:count, branch_idx] = record["next_soft_values"]
        for event_name in SAFETY_EVENTS:
            safety_arrays[event_name][:count, branch_idx] = record["events"][event_name]
        goal_events[:count, branch_idx] = record["goals"]

    measurements = compute_probe_measurements(
        rewards, log_probs, valid, terminals, truncations, next_soft_values,
        gamma=models.gamma, alpha=models.alpha, return_estimator=models.return_estimator,
        pql_horizon=pql_horizon, pql_lambda=pql_lambda,
    )
    branch_json = []
    for branch_idx, record in enumerate(branch_records):
        task = record["task"]
        q1 = record["initial_q1"]
        q2 = record["initial_q2"]
        scalar_metrics = {
            key: _scalar_at(value, branch_idx)
            for key, value in measurements.items()
            if isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == branch_count
        }
        scalar_metrics.update({
            "q1": q1,
            "q2": q2,
            "q_min": min(q1, q2),
            "initial_log_prob": record["log_probs"][0],
            "goals_reached": int(sum(record["goals"])),
            "goal_reached": bool(any(record["goals"])),
            "validity_gap": bool(record["validity_gap"]),
            "scene_boundary": bool(record["scene_boundary"]),
            "probe_horizon_reached": bool(record.get("probe_horizon_reached", False)),
            "partial_return": bool(record["validity_gap"] or not measurements["terminal"][branch_idx]),
            "repeat_idx": int(task["repeat_idx"]),
            "action_idx": int(task["action_idx"]),
            "continuation_seed": int(task["continuation_seed"]),
            "continuation_seed_scope": "policy_action_sampling_only",
            "task_id": task["task_id"],
        })
        if scalar_metrics["terminal"]:
            scalar_metrics["ending_reason"] = "terminal"
        elif scalar_metrics["probe_horizon_reached"]:
            scalar_metrics["ending_reason"] = "probe_horizon"
        elif scalar_metrics["scene_boundary"]:
            scalar_metrics["ending_reason"] = "scene_boundary"
        elif scalar_metrics["validity_gap"]:
            scalar_metrics["ending_reason"] = "validity_gap"
        elif scalar_metrics.get("truncated", False):
            scalar_metrics["ending_reason"] = "truncated"
        else:
            scalar_metrics["ending_reason"] = "censored"
        for event_name in SAFETY_EVENTS:
            scalar_metrics[event_name] = bool(np.any(safety_arrays[event_name][:, branch_idx] & valid[:, branch_idx]))
        if record["metrics_mean"] is None:
            raise RuntimeError(f"branch {task['task_id']} completed without eval metrics")
        scalar_metrics["metrics_mean"] = record["metrics_mean"]
        if record_horizons:
            scalar_metrics["horizon_metrics"] = record["horizon_metrics"]
        branch_json.append(sanitize_json(scalar_metrics))

    metric_names = sorted(branch_records[0]["metrics_mean"])
    expected_metric_names = set(metric_names)
    if any(set(record["metrics_mean"]) != expected_metric_names for record in branch_records):
        raise RuntimeError("native eval metric keys changed across branches in one probe")
    metric_array_names = [f"eval_metrics__{name.replace('/', '__')}" for name in metric_names]
    if len(set(metric_array_names)) != len(metric_array_names):
        raise ValueError("eval metric names collide after NPZ array key sanitization")

    replay_start_metadata = branch_records[0].get("replay_start_metadata")
    if replay_start_metadata is not None:
        reference_fields = {
            name: np.asarray(value).copy() for name, value in replay_start_metadata.items()
        }
    else:
        reference_fields = {
            name: np.asarray(values[probe["frame_idx"]]).copy()
            for name, values in reference.get("metadata_series", {}).items()
        }
    ego_start = {}
    for name, values in reference_fields.items():
        array = np.asarray(values)
        if array.ndim >= 1 and array.shape[0] > int(probe["ego_idx"]):
            ego_start[name] = array[int(probe["ego_idx"])]
        elif array.ndim == 0:
            ego_start[name] = array
    probe_metadata = {
        "probe_id": probe_id,
        "episode_index": int(episode_idx),
        "probe": {
            **probe,
            "active_slot": int(probe["ego_idx"]),
            "agent_id": ego_start.get("agent_id"),
            "ego_start": ego_start,
            "reference_frame_idx": int(probe["frame_idx"]),
            "reference_prefix_length": int(probe["frame_idx"]),
            "reference_fingerprint": reference["fingerprints"][probe["frame_idx"]],
            "reference_observation_fingerprint": reference["observation_fingerprints"][probe["frame_idx"]],
            "reference_file": f"references/episode_{episode_idx:03d}.npz",
            "replay_start_metadata": replay_start_metadata,
            "replay_start_fingerprint": branch_records[0].get("replay_start_fingerprint"),
            "replay_start_observation_fingerprint": branch_records[0].get(
                "replay_start_observation_fingerprint"
            ),
        },
        "scene": {
            "map_path": map_path,
            "map_sha256": map_sha,
            "scene_seed": int(scene_seed),
            "agent_count": int(agent_count),
            "reference_frame_count": int(reference["frame_count"]),
            "reference_boundary_captured": bool(reference["boundary_captured"]),
            "initial_metadata": reference["initial_metadata"],
            "metadata_fields_in_data": sorted(reference_fields),
        },
        "metrics_scope": {
            "scope": "probing_agent_forced_action_to_branch_end",
            "n": 1,
            "active_agent_count": int(agent_count),
            "agents_per_batch": sorted({
                int(record["metrics_agents_per_batch"]) for record in branch_records
            }),
            "count_semantics": "active_agent_count and agents_per_batch describe evaluation context, not ego outcomes",
            "completion_boundary": "terminal_or_truncated_or_validity_gap",
            "safety_event_flags_order": list(SAFETY_EVENTS),
            "score_definition": (
                "interval goals_reached >= configured num_goals and probing agent is not removed or stopped at interval end"
            ),
            "configured_num_goals": int(num_goals),
        },
        "metrics_schema": {
            "source": "eval_carla_fast metrics_mean",
            "branch_field": "metrics_mean",
            "array_prefix": "eval_metrics__",
            "array_dtype": "float64",
            "keys": dict(zip(metric_names, metric_array_names)),
        },
        "model": {
            "checkpoint": models.audit["checkpoint"],
            "checkpoint_sha256": models.audit["checkpoint_sha256"],
            "alpha": models.alpha,
            "gamma": models.gamma,
            "return_estimator": models.return_estimator,
            "pql_horizon": pql_horizon,
            "pql_lambda": pql_lambda,
        },
        "branches": branch_json,
    }
    arrays = {
        "rewards": rewards,
        "log_probs": log_probs,
        "valid": valid,
        "terminals": terminals,
        "truncations": truncations,
        "next_soft_values": next_soft_values,
        "goal_events": goal_events,
        "action_idx": np.asarray([record["task"]["action_idx"] for record in branch_records], dtype=np.int32),
        "repeat_idx": np.asarray([record["task"]["repeat_idx"] for record in branch_records], dtype=np.int32),
        "continuation_seed": np.asarray([record["task"]["continuation_seed"] for record in branch_records], dtype=np.int64),
        "task_id": np.asarray([record["task"]["task_id"] for record in branch_records], dtype="U160"),
        "q1": np.asarray([record["initial_q1"] for record in branch_records], dtype=np.float64),
        "q2": np.asarray([record["initial_q2"] for record in branch_records], dtype=np.float64),
    }
    arrays["initial_ego_observation"] = branch_records[0]["initial_ego_observation"]
    for event_name, values in safety_arrays.items():
        arrays[f"event_{event_name}"] = values
    for name, array_name in zip(metric_names, metric_array_names):
        arrays[array_name] = np.asarray(
            [record["metrics_mean"][name] for record in branch_records], dtype=np.float64
        )
    for name, value in reference_fields.items():
        arrays[f"scene_start__{name}"] = value
    return probe_metadata, arrays


def _scalar_at(value, branch_idx):
    item = np.asarray(value)[branch_idx]
    return item.item() if isinstance(item, np.generic) else item


def _normalize_eval_metrics(metrics):
    """Keep native eval_carla_fast fields as finite JSON/NPZ scalar values."""
    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError("native eval metrics must be a non-empty dictionary")
    normalized = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"native eval metric has invalid key: {name!r}")
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise RuntimeError(f"native eval metric {name!r} is not a numeric scalar")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(f"native eval metric {name!r} is not numeric") from error
        if not np.isfinite(scalar):
            raise RuntimeError(f"native eval metric {name!r} is not finite: {scalar!r}")
        normalized[name] = scalar
    return normalized


if __name__ == "__main__":
    main()
