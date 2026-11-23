"""Replay saved first-action probes and capture frames for the interactive viewer."""

import copy
import json
from pathlib import Path

import numpy as np

from . import models as model_module
from . import probe_env, run_probes
from .run_layout import model_directory, records_directory


def final_outcome_label(branch, arrays, branch_idx):
    """Use the final transition, since branch-level safety flags cover the entire rollout."""
    final_step = int(branch["observed_horizon"]) - 1
    if final_step < 0 or not arrays["valid"][final_step, branch_idx]:
        raise ValueError("Missing final probe transition")
    events = {name for name in run_probes.SAFETY_EVENTS if arrays[f"event_{name}"][final_step, branch_idx]}
    if branch["terminal"]:
        reasons = []
        if events & {"collision", "at_fault_collision"}:
            reasons.append("collision")
        if "offroad" in events:
            reasons.append("off-road")
        if "red_light" in events:
            reasons.append("red light")
        reason = " + ".join(reasons) or "agent terminal (cause unrecorded)"
        fault = "yes" if events & {"at_fault_collision", "offroad", "red_light"} else "no" if reasons else "unknown"
    elif branch.get("probe_horizon_reached", False):
        reason, fault = "probe horizon reached", "no"
    elif branch["scene_boundary"]:
        reason, fault = "scene ended", "no"
    elif branch["validity_gap"]:
        reason, fault = "probe became ineligible", "unknown"
    else:
        reason, fault = "rollout truncated", "unknown"
    return f"End: {reason} · Probe at fault: {fault}"


def capture(model_dir, probe_id, repeat_idx, device="cpu"):
    """Capture all twelve first-action rollouts for one saved continuation repeat.

    The original recorded reference prefix, scene, frozen checkpoint, branch
    batch sizes, and continuation seeds are reused. Replay stops at each ego's
    first terminal, truncation, validity gap, or saved scenario boundary.
    """
    model_dir = model_directory(model_dir)
    records_dir = records_directory(model_dir)
    manifest = _read_json(records_dir / "manifest.json")
    safe_probe_id = str(probe_id).replace("/", "__")
    probe_path = records_dir / "probes" / f"{safe_probe_id}.json"
    if not probe_path.is_file():
        raise FileNotFoundError(f"Probe record does not exist: {probe_path}")
    probe_record = _read_json(probe_path)
    if probe_record.get("status") != "complete" or probe_record.get("probe_id", "").replace("/", "__") != safe_probe_id:
        raise ValueError(f"Probe {probe_id!r} is not a complete record")
    if probe_record.get("run_signature") != manifest.get("signature"):
        raise ValueError("Probe run signature does not match its manifest")
    repeat_idx = int(repeat_idx)
    if repeat_idx < 0 or repeat_idx >= int(probe_record["repeats"]):
        raise ValueError(f"repeat_idx must be in [0, {probe_record['repeats']})")

    branch_json = probe_record["branches"]
    selected = {
        int(branch["action_idx"]): branch
        for branch in branch_json
        if int(branch["repeat_idx"]) == repeat_idx
    }
    if set(selected) != set(range(run_probes.NUM_ACTIONS)):
        raise ValueError("Saved probe does not contain all twelve actions for this repeat")

    arrays_path = records_dir / "probes" / probe_record["data_file"]
    if run_probes.sha256_file(arrays_path) != probe_record.get("data_sha256"):
        raise ValueError("Probe NPZ checksum does not match its JSON record")
    with np.load(arrays_path, allow_pickle=False) as saved_archive:
        saved_arrays = {key: saved_archive[key].copy() for key in saved_archive.files}

    episode_idx = int(probe_record["episode_index"])
    binned_source = probe_record.get("source", {})
    if binned_source.get("records_dir") and binned_source.get("shard"):
        source_records_dir = Path(binned_source["records_dir"]).expanduser().resolve()
        source_geometry_manifest = Path(binned_source["geometry_manifest_path"]).expanduser().resolve()
        geometry_manifest = _read_json(source_geometry_manifest)
        if geometry_manifest.get("signature") != binned_source.get("geometry_manifest_signature"):
            raise ValueError("Source geometry manifest signature differs from the probe record")
        reference_npz = (source_records_dir / binned_source["shard"]).resolve()
        try:
            reference_npz.relative_to(source_records_dir)
        except ValueError as error:
            raise ValueError("Probe reference shard escapes its source records directory") from error
        reference_json = Path(binned_source["sidecar"]).expanduser().resolve()
        if reference_json != reference_npz.with_suffix(".json"):
            raise ValueError("Probe reference sidecar does not match its source shard")
        reference_metadata = _read_json(reference_json)
        if reference_metadata.get("run_signature") != geometry_manifest.get("signature"):
            raise ValueError("Reference shard does not belong to the recorded geometry manifest")
        if run_probes.sha256_file(reference_npz) != binned_source.get("shard_sha256"):
            raise ValueError("Reference shard checksum does not match the probe record")
        if run_probes.sha256_file(reference_json) != binned_source.get("sidecar_sha256"):
            raise ValueError("Reference sidecar checksum does not match the probe record")
        if reference_metadata.get("data_file") != reference_npz.name or (
            reference_metadata.get("data_sha256") != binned_source.get("shard_sha256")
        ):
            raise ValueError("Reference shard metadata does not match its recorded identity")
        with np.load(reference_npz, allow_pickle=False) as reference_archive:
            reference_arrays = {key: reference_archive[key].copy() for key in reference_archive.files}
        env_config = dict(geometry_manifest["settings"]["resolved_env_config"])
        probe_horizon = int(manifest["run_settings"]["continuation_horizon"])
        run_seed = int(manifest["run_settings"].get("selection_seed", 20261122))
        template_seed = int(reference_metadata["layout_seed"])
        expected_model_hashes = manifest["checkpoint"].get("source_audit", {}).get("model_hashes")
    else:
        episode_npz = records_dir / "references" / f"episode_{episode_idx:03d}.npz"
        episode_json = records_dir / "references" / f"episode_{episode_idx:03d}.json"
        reference_metadata = _read_json(episode_json)
        if reference_metadata.get("run_signature") != manifest.get("signature"):
            raise ValueError("Reference run signature does not match its manifest")
        if run_probes.sha256_file(episode_npz) != reference_metadata.get("data_sha256"):
            raise ValueError("Reference NPZ checksum does not match its JSON record")
        with np.load(episode_npz, allow_pickle=False) as reference_archive:
            reference_arrays = {key: reference_archive[key].copy() for key in reference_archive.files}
        env_config = dict(manifest["resolved_config"]["resolved_env"])
        probe_horizon = None
        run_seed = int(manifest["run_settings"]["seed"])
        template_seed = run_seed
        expected_model_hashes = manifest["checkpoint"]["audit"]["model_hashes"]

    map_path = str(Path(probe_record["scene"]["map_path"]).resolve())
    if run_probes.sha256_file(map_path) != probe_record["scene"].get("map_sha256"):
        raise ValueError(f"Saved map contents differ from the recorded probe: {map_path}")
    checkpoint_path = Path(manifest["checkpoint"]["path"]).resolve()
    if run_probes.sha256_file(checkpoint_path) != manifest["checkpoint"].get("sha256"):
        raise ValueError(f"Frozen checkpoint contents differ from the recorded probe: {checkpoint_path}")
    checkpoint_config_path = Path(manifest["checkpoint"]["config_path"]).resolve()
    if run_probes.sha256_file(checkpoint_config_path) != manifest["checkpoint"].get("config_sha256"):
        raise ValueError(f"Checkpoint config contents differ from the recorded probe: {checkpoint_config_path}")
    model_config_path = checkpoint_path.with_name("config.yaml")
    if run_probes.sha256_file(model_config_path) != manifest["checkpoint"].get("config_sha256"):
        raise ValueError(f"FrozenModels config differs from the recorded checkpoint config: {model_config_path}")

    episode_seed = int(probe_record["scene"]["scene_seed"])
    agent_count = int(probe_record["scene"]["agent_count"])
    frame_idx = int(probe_record["probe"]["frame_idx"])
    ego_idx = int(probe_record["probe"]["ego_idx"])
    scenario_limit = probe_horizon if probe_horizon is not None else int(env_config["scenario_length"])
    saved_batch_size = int(manifest["run_settings"]["branch_batch_size"])
    if saved_batch_size <= 0:
        raise ValueError("Recorded branch_batch_size must be positive")

    template = probe_env.create_template(env_config, seed=template_seed)
    try:
        models = model_module.FrozenModels(checkpoint_path, template.model_env, device=device)
        if models.initial_hashes != expected_model_hashes:
            raise ValueError("Loaded actor or critic weights do not match the recorded checkpoint audit")
        tasks = run_probes.build_branch_tasks(
            str(probe_record["probe_id"]),
            run_seed,
            episode_idx,
            frame_idx,
            ego_idx,
            repeats=int(probe_record["repeats"]),
        )
        task_lookup = {task["task_id"]: task for task in tasks}
        selected_task_ids = {branch["task_id"] for branch in selected.values()}
        if not selected_task_ids.issubset(task_lookup):
            raise ValueError("Probe task IDs do not match seeds generated from the run manifest")

        captured = {}
        by_chunk = {}
        for task_idx, task in enumerate(tasks):
            if task["task_id"] in selected_task_ids:
                chunk_start = (task_idx // saved_batch_size) * saved_batch_size
                by_chunk.setdefault(chunk_start, tasks[chunk_start:chunk_start + saved_batch_size])

        initial_probabilities = None
        for chunk_tasks in by_chunk.values():
            chunk_capture, chunk_initial_probabilities = _capture_branch_chunk(
                template, models, map_path, agent_count, episode_seed,
                reference_arrays, frame_idx, ego_idx, scenario_limit,
                chunk_tasks, selected_task_ids, selected, saved_arrays, env_config,
                probe_horizon=probe_horizon, compact_prefix=binned_source.get("records_dir") is not None,
            )
            captured.update(chunk_capture)
            if initial_probabilities is None:
                initial_probabilities = chunk_initial_probabilities
            elif not np.allclose(initial_probabilities, chunk_initial_probabilities, rtol=0.0, atol=1e-7):
                raise AssertionError("Frozen actor probabilities differ across identical saved probe states")

        if set(captured) != set(range(run_probes.NUM_ACTIONS)):
            raise RuntimeError("Replay did not produce all twelve action trajectories")
        models.verify_frozen()
        return {
            "schema_version": 2,
            "probe": {
                "probe_id": str(probe_record["probe_id"]),
                "frame_idx": frame_idx,
                "recorded_repeat_idx": repeat_idx,
            },
            "initial_probabilities": initial_probabilities.tolist(),
            "actions": [captured[action_idx] for action_idx in range(run_probes.NUM_ACTIONS)],
            "recorded_repeat_idx": repeat_idx,
        }
    finally:
        template.close()


def _capture_branch_chunk(
    template, models, map_path, agent_count, scene_seed, reference_arrays,
    frame_idx, ego_idx, scenario_limit, tasks, selected_task_ids, selected,
    saved_arrays, env_config, *, probe_horizon=None, compact_prefix=False,
):
    batch = probe_env.SceneBatch.from_template(template, map_path, agent_count, len(tasks), scene_seed)
    output = {}
    try:
        batch.reset()
        native_action_table = np.asarray(batch.action_table, dtype=np.float32)
        if native_action_table.shape != models.action_table.shape or not np.allclose(
            native_action_table, models.action_table, rtol=0.0, atol=1e-6
        ):
            raise AssertionError("checkpoint and native discrete action tables do not match")
        batch.replay_prefix(
            reference_arrays["joint_actions"][:frame_idx], frame_idx,
            expected_fingerprint=str(reference_arrays["fingerprints"][frame_idx]),
            compact=compact_prefix,
        )
        observations = np.asarray(batch.observations, dtype=np.float32)
        expected_fingerprint = str(reference_arrays["observation_fingerprints"][frame_idx])
        for copy_idx in range(len(tasks)):
            if run_probes.observation_fingerprint(observations[copy_idx]) != expected_fingerprint:
                raise AssertionError("Prefix replay observation does not match the saved probing state")

        if probe_horizon is not None:
            batch.start_probe_horizon(probe_horizon)

        scenario = batch.binding.vec_get(batch.c_envs)[0]
        remap = _make_agent_remap(scenario, ego_idx)
        agent_capacity = len(scenario.get("agents") or [])
        traffic_capacity = max(1, len(scenario.get("traffic_elements") or []))
        native_frame_arrays = {
            "agent_f32": np.empty((len(tasks), agent_capacity, batch.binding.AGENT_F32_FIELDS), dtype=np.float32),
            "agent_i32": np.empty((len(tasks), agent_capacity, batch.binding.AGENT_I32_FIELDS), dtype=np.int32),
            "metrics_f32": np.empty((len(tasks), agent_capacity, batch.binding.METRICS_F32_FIELDS), dtype=np.float32),
            "puffer_f32": np.empty((len(tasks), agent_capacity, batch.binding.SCORE_F32_FIELDS), dtype=np.float32),
            "traffic_i16": np.empty((len(tasks), traffic_capacity, batch.binding.TRAFFIC_I16_FIELDS), dtype=np.int16),
        }
        frame_records = {}
        for copy_idx, task in enumerate(tasks):
            if task["task_id"] in selected_task_ids:
                frame_records[copy_idx] = {"rewards": []}

        rngs = batch.set_continuation_seeds([task["continuation_seed"] for task in tasks])
        live = np.ones(len(tasks), dtype=np.bool_)
        selected_copy_indices = {
            copy_idx: int(task["action_idx"])
            for copy_idx, task in enumerate(tasks)
            if task["task_id"] in selected_task_ids
        }
        initial_probabilities = None
        pool_slot_counts = getattr(models.actor, "pool_slot_counts", None)
        for time_idx in range(scenario_limit):
            if not any(live[copy_idx] for copy_idx in selected_copy_indices):
                break
            current_observations = np.asarray(batch.observations, dtype=np.float32)
            flat_observations = current_observations.reshape(len(tasks) * agent_count, -1)
            probabilities_flat, _ = models.policy(flat_observations)
            probabilities = probabilities_flat.reshape(len(tasks), agent_count, run_probes.NUM_ACTIONS)
            if time_idx == 0:
                initial_probabilities = probabilities[0, ego_idx].copy()
                for copy_idx in range(1, len(tasks)):
                    if not np.allclose(probabilities[copy_idx, ego_idx], initial_probabilities, rtol=0.0, atol=1e-7):
                        raise AssertionError("Identical saved probe observations produced different frozen actor probabilities")
            discrete_actions = np.zeros((len(tasks), agent_count), dtype=np.int32)
            for copy_idx in np.flatnonzero(live):
                discrete_actions[copy_idx] = run_probes.sample_actions(probabilities[copy_idx], rngs[copy_idx])
            if time_idx == 0:
                discrete_actions[np.arange(len(tasks)), ego_idx] = np.asarray(
                    [int(task["action_idx"]) for task in tasks], dtype=np.int32
                )

            batch.binding.vec_get_obs_html_frame(
                batch.c_envs,
                native_frame_arrays["agent_f32"],
                native_frame_arrays["agent_i32"],
                native_frame_arrays["metrics_f32"],
                native_frame_arrays["puffer_f32"],
                native_frame_arrays["traffic_i16"],
            )
            pool_frames = _capture_pool_slot_counts(
                pool_slot_counts, flat_observations, len(tasks), agent_count, models
            )
            for copy_idx in np.flatnonzero(live):
                if copy_idx not in selected_copy_indices:
                    continue
                _append_replay_frame(
                    frame_records[copy_idx],
                    native_frame_arrays,
                    copy_idx,
                    current_observations[copy_idx],
                    probabilities[copy_idx],
                    discrete_actions[copy_idx],
                    models.action_table,
                    pool_frames,
                    remap,
                )
            step_result = batch.step(
                models.action_table[discrete_actions], compact_metadata=probe_horizon is not None
            )
            rewards = np.asarray(step_result["rewards"], dtype=np.float32).reshape(len(tasks), agent_count)
            terminals = np.asarray(step_result["terminals"], dtype=np.bool_).reshape(len(tasks), agent_count)
            truncations = np.asarray(step_result["truncations"], dtype=np.bool_).reshape(len(tasks), agent_count)
            metadata = step_result["metadata"]
            scene_done = np.asarray(metadata["scene_done"], dtype=np.bool_).reshape(len(tasks))
            for copy_idx in np.flatnonzero(live):
                terminal = bool(terminals[copy_idx, ego_idx])
                truncated = bool(truncations[copy_idx, ego_idx])
                if (
                    copy_idx in selected_copy_indices
                    and scene_done[copy_idx]
                    and not terminal
                    and not truncated
                ):
                    raise RuntimeError(
                        "native scene boundary failed to mark the probe ego terminal or truncated"
                    )
                validity_gap = not bool(metadata["eligible"][copy_idx, ego_idx]) and not terminal and not truncated
                done = terminal or truncated or validity_gap
                action_idx = selected_copy_indices.get(int(copy_idx))
                if action_idx is not None:
                    frame_records[copy_idx]["rewards"].append(float(rewards[copy_idx, ego_idx]))
                if done:
                    live[copy_idx] = False
                    if action_idx is not None:
                        stop_reason = "terminal" if terminal else "truncated" if truncated else "validity_gap"
                        saved_branch = selected[action_idx]
                        saved_repeat_idx = int(saved_branch["repeat_idx"])
                        indices = np.flatnonzero(
                            (saved_arrays["repeat_idx"] == saved_repeat_idx)
                            & (saved_arrays["action_idx"] == action_idx)
                        )
                        if indices.size != 1:
                            raise ValueError("Saved branch arrays do not uniquely identify action/repeat")
                        branch_position = int(indices[0])
                        expected_valid = saved_arrays["valid"][:, branch_position].astype(bool)
                        actual_rewards = np.asarray(frame_records[copy_idx]["rewards"], dtype=np.float32)
                        expected_rewards = saved_arrays["rewards"][:, branch_position][expected_valid].astype(np.float32)
                        if actual_rewards.shape != expected_rewards.shape or not np.allclose(
                            actual_rewards, expected_rewards, rtol=0.0, atol=2e-5
                        ):
                            shared_count = min(actual_rewards.size, expected_rewards.size)
                            differing_indices = np.flatnonzero(
                                np.abs(actual_rewards[:shared_count] - expected_rewards[:shared_count]) > 2e-5
                            )
                            first_difference = int(differing_indices[0]) if differing_indices.size else None
                            raise RuntimeError(
                                f"Replay diverged from saved rewards for action {action_idx}, repeat {saved_repeat_idx}: "
                                f"captured={actual_rewards.size}, saved={expected_rewards.size}, "
                                f"first_difference={first_difference}, "
                                f"captured_reward={actual_rewards[first_difference] if first_difference is not None else None}, "
                                f"saved_reward={expected_rewards[first_difference] if first_difference is not None else None}"
                            )
                        if len(actual_rewards) != int(saved_branch["observed_horizon"]):
                            raise RuntimeError(
                                f"Replay horizon {len(actual_rewards)} differs from recorded horizon "
                                f"{saved_branch['observed_horizon']} for action {action_idx}"
                            )
                        output[action_idx] = {
                            "action_idx": action_idx,
                            "action_label": _action_label(action_idx, models.action_table_physical),
                            "initial_probability": float(initial_probabilities[action_idx]),
                            "observed_horizon": len(actual_rewards),
                            "q_min": saved_branch["q_min"],
                            "actual_critic_target": saved_branch["actual_critic_target"],
                            "recorded_return": float(saved_branch["discounted_reward_return"]),
                            "stop_reason": stop_reason,
                            "final_outcome": final_outcome_label(saved_branch, saved_arrays, branch_position),
                            "replay_bytes": _encode_action_replay(
                                scenario,
                                frame_records[copy_idx],
                                env_config,
                                remap,
                                int(template.probe_env_kwargs["init_step"]),
                                frame_idx,
                            ),
                        }
                        del frame_records[copy_idx]
            if not any(live[copy_idx] for copy_idx in selected_copy_indices):
                break
        if any(live[copy_idx] for copy_idx in selected_copy_indices):
            raise RuntimeError("Rollout reached scenario limit before a recorded terminal/truncation")
        if set(output) != set(selected_copy_indices.values()):
            raise RuntimeError("Replay did not encode every selected action")
        return output, initial_probabilities
    finally:
        batch.close()


def _make_agent_remap(scenario, ego_active_idx):
    agents = scenario.get("agents") or []
    active_indices = scenario.get("active_agent_indices") or []
    if not 0 <= ego_active_idx < len(active_indices):
        raise RuntimeError(f"Probing active-agent slot {ego_active_idx} is missing from the replay scenario")
    agent_capacity = len(agents)
    ego_agent_idx = int(active_indices[ego_active_idx])
    if not 0 <= ego_agent_idx < agent_capacity:
        raise RuntimeError(f"Probing native agent index {ego_agent_idx} is outside the scenario agent array")
    agent_order = [ego_agent_idx] + [idx for idx in range(agent_capacity) if idx != ego_agent_idx]
    agent_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(agent_order)}
    active_order = [ego_active_idx] + [idx for idx in range(len(active_indices)) if idx != ego_active_idx]
    active_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(active_order)}
    return {
        "agent_order": agent_order,
        "agent_remap": agent_remap,
        "active_order": active_order,
        "active_remap": active_remap,
        "old_active_indices": [int(idx) for idx in active_indices],
    }


def _remap_scenario(scenario, remap):
    remapped = copy.deepcopy(scenario)
    remapped["agents"] = [copy.deepcopy(scenario["agents"][old_idx]) for old_idx in remap["agent_order"]]
    for new_idx, agent in enumerate(remapped["agents"]):
        agent["id"] = new_idx
    remapped["active_agent_indices"] = [
        remap["agent_remap"][remap["old_active_indices"][old_slot]]
        for old_slot in remap["active_order"]
    ]
    remapped["static_agent_indices"] = [
        remap["agent_remap"][int(idx)] for idx in (scenario.get("static_agent_indices") or [])
    ]
    for key in ("objects_of_interest", "tracks_to_predict"):
        values = scenario.get(key)
        if values is not None:
            remapped[key] = [remap["agent_remap"][int(idx)] for idx in values]
    return remapped


def _capture_pool_slot_counts(pool_slot_counts, flat_observations, copy_count, agent_count, models):
    if pool_slot_counts is None:
        return {}
    import torch

    observation_tensor = torch.as_tensor(flat_observations, device=models.device)
    with torch.inference_mode():
        count_values = pool_slot_counts(observation_tensor)
    return {
        name: values.detach().reshape(copy_count, agent_count, -1).cpu().numpy().astype(np.int16, copy=False)
        for name, values in count_values.items()
    }


def _append_replay_frame(
    frames, native_arrays, copy_idx, observations, probabilities, discrete_actions,
    action_table, pool_frames, remap,
):
    for name in ("agent_f32", "agent_i32", "metrics_f32", "puffer_f32"):
        frame = native_arrays[name][copy_idx, remap["agent_order"]].copy()
        if name == "agent_i32":
            frame[:, 0] = np.arange(len(remap["agent_order"]), dtype=np.int32)
            frame[:, 7] = -1
            for old_slot, old_agent_idx in enumerate(remap["old_active_indices"]):
                new_agent_idx = remap["agent_remap"][old_agent_idx]
                frame[new_agent_idx, 7] = remap["active_remap"][old_slot]
        frames.setdefault(name, []).append(frame)
    frames.setdefault("traffic_i16", []).append(native_arrays["traffic_i16"][copy_idx].copy())
    action_order = remap["active_order"]
    action_indices = np.asarray(discrete_actions[action_order], dtype=np.int32)
    frames.setdefault("raw_action", []).append(action_indices.astype(np.float32)[:, None])
    frames.setdefault("clipped_action", []).append(np.asarray(action_table[action_indices], dtype=np.float32))
    ordered_probabilities = np.asarray(probabilities[action_order], dtype=np.float32)
    frames.setdefault("policy_probs", []).append(ordered_probabilities.copy())
    entropy = -np.sum(ordered_probabilities * np.log(np.maximum(ordered_probabilities, 1e-30)), axis=-1)
    frames.setdefault("entropy", []).append(entropy.astype(np.float32))
    frames.setdefault("obs", []).append(np.asarray(observations[action_order], dtype=np.float16).copy())
    for name, values in pool_frames.items():
        frames.setdefault(name, []).append(values[copy_idx, action_order].copy())


def _encode_action_replay(scenario, frame_records, env_config, remap, initial_step, frame_idx):
    from pufferlib.sac_viz import encode_interactive_replay

    replay_env_config = dict(env_config)
    replay_env_config["init_step"] = int(initial_step) + int(frame_idx)
    replay = {"env": replay_env_config}
    for name in list(frame_records):
        if name == "rewards":
            continue
        frames = frame_records.pop(name)
        replay[name] = np.stack(frames, axis=0)
        del frames
    return encode_interactive_replay(_remap_scenario(scenario, remap), replay)


def _action_label(action_idx, action_table_physical):
    longitudinal_jerk, lateral_jerk = action_table_physical[action_idx]
    return (
        f"action {action_idx}: longitudinal jerk {float(longitudinal_jerk):g} m/s³, "
        f"lateral jerk {float(lateral_jerk):g} m/s³"
    )


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read saved calibration JSON {path}: {error}") from error
