"""Collect 12-action, frozen-PPO probes from selected CARLA self-play states.

Each branch runs for up to 2560 transitions so its driving outcomes match the
SAC probes.  The PPO value target is reconstructed from only the first
128 learner slots, using the same row alignment and native GAE/V-trace kernel
as training.
"""

from collections import defaultdict
from pathlib import Path
import json
import sys

import numpy as np

from . import collect_binned_probes as binned
from .run_layout import initialize_model_directory, model_directory, records_directory
from .run_probes import (
    NUM_ACTIONS,
    SAFETY_EVENTS,
    _metadata_copy,
    _native_build_fingerprint,
    _normalize_eval_metrics,
    _scene_parameters,
    atomic_write_json,
    build_branch_tasks,
    current_utc,
    observation_fingerprint,
    open_or_create_manifest,
    probe_is_complete,
    probe_paths,
    sample_actions,
    sanitize_json,
    save_probe,
    sha256_file,
    source_hashes,
    stable_seed,
)


PROBE_HORIZON = 2560
PPO_SEGMENT_ROWS = 128
PPO_SEGMENT_TRANSITIONS = PPO_SEGMENT_ROWS - 1
REPEATS = 1
PPO_COLLECTOR_FORMAT_VERSION = 1


def build_argument_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Destination model directory under runs/<run>/ppo")
    parser.add_argument("--source-model-dir", type=Path, required=True,
                        help="PPO geometry model directory under runs/second_*/ppo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    return parser


def _reward_for_ppo(reward, normalize_rewards):
    reward = np.asarray(reward, dtype=np.float32)
    if not np.isfinite(reward).all():
        raise ValueError("PPO rewards must be finite")
    if not normalize_rewards:
        return reward.copy()
    return (np.sign(reward) * np.log1p(np.abs(reward))).astype(np.float32)


def _native_ppo_advantages(values, rewards, dones, importance, gamma, gae_lambda,
                           rho_clip=1.0, c_clip=1.0):
    """Run the exact CPU kernel used by pufferl over one aligned segment."""
    import torch

    arrays = [np.asarray(item, dtype=np.float32) for item in (values, rewards, dones, importance)]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("PPO kernel inputs must be one-dimensional slot arrays")
    if len({array.size for array in arrays}) != 1 or arrays[0].size < 2:
        raise ValueError("PPO kernel input arrays must have a shared length of at least two")
    values, rewards, dones, importance = arrays
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("PPO kernel inputs must be finite")
    if np.any((dones != 0.0) & (dones != 1.0)):
        raise ValueError("PPO done slots must contain only zero or one")
    if not 0.0 <= float(gamma) <= 1.0 or not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("PPO gamma and gae_lambda must be in [0, 1]")

    # Import only the extension, not pufferl (which installs process signal handlers).
    import pufferlib._C  # noqa: F401

    value_tensor = torch.from_numpy(np.ascontiguousarray(values[None, :]))
    reward_tensor = torch.from_numpy(np.ascontiguousarray(rewards[None, :]))
    done_tensor = torch.from_numpy(np.ascontiguousarray(dones[None, :]))
    importance_tensor = torch.from_numpy(np.ascontiguousarray(importance[None, :]))
    advantage_tensor = torch.zeros_like(value_tensor)
    torch.ops.pufferlib.compute_puff_advantage(
        value_tensor, reward_tensor, done_tensor, importance_tensor, advantage_tensor,
        float(gamma), float(gae_lambda), float(rho_clip), float(c_clip),
    )
    return advantage_tensor.numpy()[0].copy()


def prepare_ppo_segment(
    state_values,
    transition_rewards,
    terminals,
    truncations,
    learner_masks,
    *,
    gamma,
    gae_lambda,
    normalize_rewards=False,
    use_value_bootstrapping=True,
    rho_clip=1.0,
    c_clip=1.0,
):
    """Build the 128-slot training alignment and exact first-slot PPO target.

    Action-row rollout arrays record the reward caused by action t at row t.
    Puffer's learner instead stores it with the destination observation in row
    t+1.  This helper prepends the probe observation/value and a zero reward.
    A physical terminal, truncation, or zero learner mask stops the recurrence.
    """
    state_values = np.asarray(state_values, dtype=np.float32)
    transition_rewards = np.asarray(transition_rewards, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=np.bool_)
    truncations = np.asarray(truncations, dtype=np.bool_)
    learner_masks = np.asarray(learner_masks, dtype=np.bool_)
    transition_count = transition_rewards.size
    if transition_count < 1:
        raise ValueError("PPO target requires at least one observed transition")
    if any(array.ndim != 1 for array in (state_values, transition_rewards, terminals,
                                          truncations, learner_masks)):
        raise ValueError("PPO segment inputs must be one-dimensional")
    if terminals.size != transition_count or truncations.size != transition_count or learner_masks.size != transition_count:
        raise ValueError("PPO transition rewards and boundary arrays must have matching lengths")
    if state_values.size != transition_count + 1:
        raise ValueError("state_values must include the initial state and one destination slot per transition")
    if not np.isfinite(state_values).all() or not np.isfinite(transition_rewards).all():
        raise ValueError("physical PPO state values and rewards must be finite")

    boundary = terminals | truncations | ~learner_masks
    first_boundary = np.flatnonzero(boundary)
    target_transition_count = min(PPO_SEGMENT_TRANSITIONS, transition_count)
    boundary_type = "segment_limit"
    if first_boundary.size and int(first_boundary[0]) < target_transition_count:
        target_transition_count = int(first_boundary[0]) + 1
        boundary_idx = target_transition_count - 1
        if terminals[boundary_idx]:
            boundary_type = "terminal"
        elif truncations[boundary_idx]:
            boundary_type = "truncation"
        else:
            boundary_type = "learner_mask_gap"

    row_count = target_transition_count + 1
    aligned_values = np.zeros(PPO_SEGMENT_ROWS, dtype=np.float32)
    aligned_rewards = np.zeros(PPO_SEGMENT_ROWS, dtype=np.float32)
    aligned_dones = np.zeros(PPO_SEGMENT_ROWS, dtype=np.float32)
    aligned_importance = np.ones(PPO_SEGMENT_ROWS, dtype=np.float32)
    aligned_masks = np.zeros(PPO_SEGMENT_ROWS, dtype=np.bool_)
    aligned_truncations = np.zeros(PPO_SEGMENT_ROWS, dtype=np.bool_)
    aligned_value_valid = np.zeros(PPO_SEGMENT_ROWS, dtype=np.bool_)
    slot_valid = np.zeros(PPO_SEGMENT_ROWS, dtype=np.bool_)

    aligned_values[:target_transition_count] = state_values[:target_transition_count]
    aligned_value_valid[:target_transition_count] = True
    aligned_rewards[1:row_count] = _reward_for_ppo(
        transition_rewards[:target_transition_count], normalize_rewards
    )
    aligned_masks[:row_count] = True
    aligned_truncations[1:row_count] = truncations[:target_transition_count]
    aligned_dones[1:row_count] = boundary[:target_transition_count].astype(np.float32)
    slot_valid[:row_count] = True

    final_boundary_idx = target_transition_count - 1
    boundary_reached = bool(boundary[final_boundary_idx])
    if not boundary_reached:
        # A 128-row learner segment bootstraps at the actual state s_127.
        aligned_values[target_transition_count] = state_values[target_transition_count]
        aligned_value_valid[target_transition_count] = True
    else:
        # The destination value is masked by done[t+1]; no post-reset
        # observation is required or included in this trajectory.
        aligned_values[target_transition_count] = 0.0
        aligned_value_valid[target_transition_count] = False
        if not learner_masks[final_boundary_idx]:
            aligned_masks[target_transition_count] = False
        if (
            truncations[final_boundary_idx]
            and not terminals[final_boundary_idx]
            and use_value_bootstrapping
        ):
            # Match pufferl.py's autoreset truncation proxy exactly.
            aligned_rewards[target_transition_count] += (
                np.float32(gamma) * state_values[target_transition_count - 1]
            )

    advantages = _native_ppo_advantages(
        aligned_values, aligned_rewards, aligned_dones, aligned_importance,
        gamma, gae_lambda, rho_clip, c_clip,
    )
    target = np.float32(aligned_values[0] + advantages[0])
    return {
        "ppo_values": aligned_values,
        "ppo_training_rewards": aligned_rewards,
        "ppo_training_dones": aligned_dones,
        "ppo_importance": aligned_importance,
        "ppo_advantages": advantages,
        "ppo_training_masks": aligned_masks,
        "ppo_training_truncations": aligned_truncations,
        "ppo_value_valid": aligned_value_valid,
        "ppo_slot_valid": slot_valid,
        "ppo_advantage": float(advantages[0]),
        "initial_value": float(aligned_values[0]),
        "ppo_value_target": float(target),
        "ppo_segment_rows": int(row_count),
        "ppo_segment_transitions": int(target_transition_count),
        "ppo_segment_boundary": boundary_type,
        "ppo_target_status": "native_kernel_exact_for_observed_branch",
    }


def _policy_and_values(models, observations):
    """Perform one frozen-model forward pass for categorical policy and V."""
    import torch

    logits, values = models._forward(observations)
    log_probs = torch.log_softmax(logits, dim=-1)
    return (
        log_probs.exp().cpu().numpy(),
        log_probs.cpu().numpy(),
        values.cpu().numpy(),
    )


def _run_branch_batch(scene_batch_type, template, map_path, agent_count, scene_seed,
                      reference, probe, tasks, models, progress_callback=None):
    import torch

    copy_count = len(tasks)
    frame_idx = int(probe["frame_idx"])
    ego_idx = int(probe["ego_idx"])
    batch = scene_batch_type.from_template(template, map_path, agent_count, copy_count, scene_seed)
    records = [
        {
            "task": task,
            "rewards": [], "training_rewards": [], "log_probs": [], "actions": [],
            "terminals": [], "truncations": [], "learner_masks": [], "state_values": [],
            "state_value_valid": [], "valid": [],
            "events": {name: [] for name in SAFETY_EVENTS}, "goals": [],
            "validity_gap": False, "scene_boundary": False, "probe_horizon_reached": False,
            "metrics_mean": None,
        }
        for task in tasks
    ]
    try:
        batch.reset()
        if not np.allclose(batch.action_table, models.action_table, rtol=0.0, atol=1e-7):
            raise AssertionError("PPO checkpoint action table differs from native jerk action order")
        batch.replay_prefix(
            reference["joint_actions"][:frame_idx], frame_idx,
            expected_fingerprint=reference["fingerprints"][frame_idx], compact=True,
        )
        observations = np.asarray(batch.observations, dtype=np.float32)
        expected_observation_fingerprint = reference["observation_fingerprints"][frame_idx]
        if any(
            observation_fingerprint(observations[copy_idx]) != expected_observation_fingerprint
            for copy_idx in range(copy_count)
        ):
            raise AssertionError("replayed PPO branch observation differs from geometry reference")
        rngs = batch.set_continuation_seeds([task["continuation_seed"] for task in tasks])
        replay_start_metadata = _metadata_copy(batch.metadata(), 0)
        replay_start_fingerprint = batch.fingerprint(0)
        replay_start_observation_fingerprint = observation_fingerprint(observations[0])
        if any(batch.fingerprint(copy_idx) != replay_start_fingerprint for copy_idx in range(1, copy_count)):
            raise AssertionError("branch copies diverged from the replayed initial scene")
        batch.start_probe_horizon(PROBE_HORIZON)
        for record in records:
            record["replay_start_metadata"] = replay_start_metadata
            record["replay_start_fingerprint"] = replay_start_fingerprint
            record["replay_start_observation_fingerprint"] = replay_start_observation_fingerprint

        flat_observations = observations.reshape(copy_count * agent_count, -1)
        probabilities_flat, log_probs_flat, values_flat = _policy_and_values(models, flat_observations)
        probabilities = probabilities_flat.reshape(copy_count, agent_count, NUM_ACTIONS)
        log_probs = log_probs_flat.reshape(copy_count, agent_count, NUM_ACTIONS)
        values = values_flat.reshape(copy_count, agent_count)
        initial_probabilities = probabilities[0, ego_idx].copy()
        initial_value = float(values[0, ego_idx])
        for copy_idx in range(copy_count):
            if not np.allclose(probabilities[copy_idx, ego_idx], initial_probabilities, rtol=0.0, atol=1e-7):
                raise AssertionError("identical PPO probe observations produced different actor probabilities")
            if not np.isclose(values[copy_idx, ego_idx], initial_value, rtol=0.0, atol=1e-6):
                raise AssertionError("identical PPO probe observations produced different values")
            records[copy_idx]["initial_ego_observation"] = observations[copy_idx, ego_idx].copy()
            records[copy_idx]["initial_probabilities"] = initial_probabilities.copy()

        metrics_baseline = batch.begin_metrics(ego_idx)
        event_flags = np.zeros((copy_count, len(SAFETY_EVENTS)), dtype=np.bool_)
        live = np.ones(copy_count, dtype=np.bool_)
        action_limit = PROBE_HORIZON
        normalize_rewards = bool(models.train_config.get("normalize_rewards", False))
        for time_idx in range(action_limit):
            if not np.any(live):
                break
            current_observations = np.asarray(batch.observations, dtype=np.float32)
            flat_observations = current_observations.reshape(copy_count * agent_count, -1)
            probabilities_flat, log_probs_flat, values_flat = _policy_and_values(models, flat_observations)
            probabilities = probabilities_flat.reshape(copy_count, agent_count, NUM_ACTIONS)
            log_probs = log_probs_flat.reshape(copy_count, agent_count, NUM_ACTIONS)
            values = values_flat.reshape(copy_count, agent_count)
            discrete_actions = np.zeros((copy_count, agent_count), dtype=np.int32)
            for copy_idx in np.flatnonzero(live):
                discrete_actions[copy_idx] = sample_actions(probabilities[copy_idx], rngs[copy_idx])
                record = records[copy_idx]
                record["state_values"].append(float(values[copy_idx, ego_idx]))
                record["state_value_valid"].append(True)
            if time_idx == 0:
                discrete_actions[np.arange(copy_count), ego_idx] = np.asarray(
                    [int(task["action_idx"]) for task in tasks], dtype=np.int32
                )
            ego_action_ids = discrete_actions[:, ego_idx].copy()
            ego_log_probs = log_probs[np.arange(copy_count), ego_idx, ego_action_ids]
            step_result = batch.step(models.action_table[discrete_actions], compact_metadata=True)
            step_rewards = np.asarray(step_result["rewards"], dtype=np.float32).reshape(copy_count, agent_count)
            terminals = np.asarray(step_result["terminals"], dtype=np.bool_).reshape(copy_count, agent_count)
            truncations = np.asarray(step_result["truncations"], dtype=np.bool_).reshape(copy_count, agent_count)
            metadata = step_result["metadata"]
            events = step_result["events"]
            for event_name in (*SAFETY_EVENTS, "reached_goal"):
                if event_name not in events:
                    raise RuntimeError(f"native branch step omitted event flag {event_name!r}")
                events[event_name] = np.asarray(events[event_name], dtype=np.bool_).reshape(copy_count, agent_count)
            post_eligible = np.asarray(metadata["eligible"], dtype=np.bool_).reshape(copy_count, agent_count)
            scene_done = np.asarray(metadata["scene_done"], dtype=np.bool_).reshape(copy_count)
            completed_indices = []
            for copy_idx in np.flatnonzero(live):
                record = records[copy_idx]
                terminal = bool(terminals[copy_idx, ego_idx])
                truncated = bool(truncations[copy_idx, ego_idx])
                boundary = bool(scene_done[copy_idx])
                if boundary and not terminal and not truncated:
                    raise RuntimeError("native scene boundary did not mark PPO probe ego terminal or truncated")
                learner_mask = bool(post_eligible[copy_idx, ego_idx])
                validity_gap = not learner_mask and not terminal and not truncated
                reward = float(step_rewards[copy_idx, ego_idx])
                record["rewards"].append(reward)
                record["training_rewards"].append(float(_reward_for_ppo(reward, normalize_rewards)))
                record["log_probs"].append(float(ego_log_probs[copy_idx]))
                record["actions"].append(int(ego_action_ids[copy_idx]))
                record["terminals"].append(terminal)
                record["truncations"].append(truncated)
                record["learner_masks"].append(learner_mask)
                record["valid"].append(True)
                record["validity_gap"] = record["validity_gap"] or validity_gap
                record["scene_boundary"] = record["scene_boundary"] or boundary
                record["probe_horizon_reached"] = bool(
                    truncated and time_idx + 1 == PROBE_HORIZON
                )
                for event_idx, event_name in enumerate(SAFETY_EVENTS):
                    event = bool(events[event_name][copy_idx, ego_idx])
                    record["events"][event_name].append(event)
                    event_flags[copy_idx, event_idx] |= event
                record["goals"].append(bool(events["reached_goal"][copy_idx, ego_idx]))
                done = terminal or truncated or validity_gap
                if done:
                    record["state_values"].append(0.0)
                    record["state_value_valid"].append(False)
                    live[copy_idx] = False
                    completed_indices.append(int(copy_idx))

            if completed_indices:
                metrics_by_copy = batch.metrics(metrics_baseline, event_flags)
                if not isinstance(metrics_by_copy, (list, tuple)) or len(metrics_by_copy) != copy_count:
                    raise RuntimeError("native probe metrics did not return one row per branch")
                for copy_idx in completed_indices:
                    records[copy_idx]["metrics_mean"] = _normalize_eval_metrics(metrics_by_copy[copy_idx])

            if np.any(scene_done & live):
                raise RuntimeError("native scene boundary failed to stop a live PPO probe branch")
            if progress_callback is not None:
                progress_callback(time_idx + 1, int(live.sum()))

        if np.any(live):
            raise RuntimeError(f"PPO branch continuation exceeded configured horizon={action_limit}")
        for record in records:
            if record["metrics_mean"] is None:
                raise RuntimeError(f"PPO branch {record['task']['task_id']} ended without eval metrics")
            if len(record["state_values"]) != len(record["valid"]) + 1:
                raise RuntimeError("PPO value sequence must contain one value per state boundary")
        return records
    finally:
        batch.close()


def _discounted_sum(values, gamma):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("discounted return requires a non-empty finite reward sequence")
    discounts = np.power(float(gamma), np.arange(values.size, dtype=np.float64))
    return float(np.dot(discounts, values))


def _assemble_probe_bundle(probe_id, episode_idx, map_path, map_sha, scene_seed, reference,
                           probe, agent_count, branch_records, models, *, num_goals):
    branch_count = len(branch_records)
    observed_horizon = max(len(record["valid"]) for record in branch_records)
    shape = (observed_horizon, branch_count)
    rewards = np.full(shape, np.nan, dtype=np.float32)
    training_rewards = np.full(shape, np.nan, dtype=np.float32)
    log_probs = np.full(shape, np.nan, dtype=np.float32)
    valid = np.zeros(shape, dtype=np.bool_)
    terminals = np.zeros(shape, dtype=np.bool_)
    truncations = np.zeros(shape, dtype=np.bool_)
    ego_action_idx = np.full(shape, -1, dtype=np.int32)
    values = np.full((observed_horizon + 1, branch_count), np.nan, dtype=np.float32)
    value_valid = np.zeros((observed_horizon + 1, branch_count), dtype=np.bool_)
    safety_arrays = {name: np.zeros(shape, dtype=np.bool_) for name in SAFETY_EVENTS}
    goal_events = np.zeros(shape, dtype=np.bool_)
    ppo_arrays = {
        name: np.zeros((PPO_SEGMENT_ROWS, branch_count), dtype=dtype)
        for name, dtype in (
            ("ppo_values", np.float32),
            ("ppo_training_rewards", np.float32),
            ("ppo_training_dones", np.float32),
            ("ppo_importance", np.float32),
            ("ppo_advantages", np.float32),
            ("ppo_training_masks", np.bool_),
            ("ppo_training_truncations", np.bool_),
            ("ppo_value_valid", np.bool_),
            ("ppo_slot_valid", np.bool_),
        )
    }
    ppo_arrays["ppo_importance"].fill(1.0)
    branches = []
    for branch_idx, record in enumerate(branch_records):
        count = len(record["valid"])
        rewards[:count, branch_idx] = record["rewards"]
        training_rewards[:count, branch_idx] = record["training_rewards"]
        log_probs[:count, branch_idx] = record["log_probs"]
        valid[:count, branch_idx] = True
        terminals[:count, branch_idx] = record["terminals"]
        truncations[:count, branch_idx] = record["truncations"]
        ego_action_idx[:count, branch_idx] = record["actions"]
        values[:count + 1, branch_idx] = record["state_values"]
        value_valid[:count + 1, branch_idx] = record["state_value_valid"]
        for name in SAFETY_EVENTS:
            safety_arrays[name][:count, branch_idx] = record["events"][name]
        goal_events[:count, branch_idx] = record["goals"]

        segment = prepare_ppo_segment(
            record["state_values"], record["rewards"], record["terminals"],
            record["truncations"], record["learner_masks"],
            gamma=models.gamma,
            gae_lambda=float(models.train_config["gae_lambda"]),
            normalize_rewards=bool(models.train_config.get("normalize_rewards", False)),
            use_value_bootstrapping=bool(models.train_config.get("use_value_bootstrapping", True)),
            rho_clip=float(models.train_config["vtrace_rho_clip"]),
            c_clip=float(models.train_config["vtrace_c_clip"]),
        )
        for name in ppo_arrays:
            ppo_arrays[name][:, branch_idx] = segment[name]

        target = segment["ppo_value_target"]
        raw_discounted_return = _discounted_sum(record["rewards"], models.gamma)
        transformed_discounted_return = _discounted_sum(record["training_rewards"], models.gamma)
        has_terminal = bool(any(record["terminals"]))
        has_truncation = bool(any(record["truncations"]))
        branch = {
            "initial_probability": float(record["initial_probabilities"][record["task"]["action_idx"]]),
            "initial_log_prob": float(record["log_probs"][0]),
            "initial_value": segment["initial_value"],
            "ppo_advantage": segment["ppo_advantage"],
            "ppo_value_target": target,
            "ppo_target_status": segment["ppo_target_status"],
            "ppo_segment_rows": segment["ppo_segment_rows"],
            "ppo_segment_transitions": segment["ppo_segment_transitions"],
            "ppo_segment_boundary": segment["ppo_segment_boundary"],
            "discounted_reward_return": raw_discounted_return,
            "discounted_ppo_reward_return": transformed_discounted_return,
            "observed_horizon": int(count),
            "terminal": has_terminal,
            "truncated": has_truncation,
            "validity_gap": bool(record["validity_gap"]),
            "scene_boundary": bool(record["scene_boundary"]),
            "probe_horizon_reached": bool(record["probe_horizon_reached"]),
            "partial_return": bool(record["validity_gap"] or not has_terminal),
            "goals_reached": int(sum(record["goals"])),
            "goal_reached": bool(any(record["goals"])),
            "repeat_idx": int(record["task"]["repeat_idx"]),
            "action_idx": int(record["task"]["action_idx"]),
            "continuation_seed": int(record["task"]["continuation_seed"]),
            "continuation_seed_scope": "policy_action_sampling_only",
            "task_id": record["task"]["task_id"],
            "metrics_mean": record["metrics_mean"],
        }
        for event_name in SAFETY_EVENTS:
            branch[event_name] = bool(np.any(safety_arrays[event_name][:, branch_idx]))
        if has_terminal:
            branch["ending_reason"] = "terminal"
        elif record["probe_horizon_reached"]:
            branch["ending_reason"] = "probe_horizon"
        elif record["scene_boundary"]:
            branch["ending_reason"] = "scene_boundary"
        elif record["validity_gap"]:
            branch["ending_reason"] = "validity_gap"
        elif has_truncation:
            branch["ending_reason"] = "truncated"
        else:
            branch["ending_reason"] = "censored"
        branches.append(sanitize_json(branch))

    metric_names = sorted(branch_records[0]["metrics_mean"])
    if any(set(record["metrics_mean"]) != set(metric_names) for record in branch_records):
        raise RuntimeError("native evaluation metric keys changed across PPO branches")
    metric_array_names = [f"eval_metrics__{name.replace('/', '__')}" for name in metric_names]
    if len(set(metric_array_names)) != len(metric_array_names):
        raise ValueError("native evaluation metric names collide after NPZ key sanitization")
    replay_start_metadata = branch_records[0]["replay_start_metadata"]
    reference_fields = {name: np.asarray(value).copy() for name, value in replay_start_metadata.items()}
    ego_start = {}
    for name, value in reference_fields.items():
        array = np.asarray(value)
        if array.ndim and array.shape[0] > int(probe["ego_idx"]):
            ego_start[name] = array[int(probe["ego_idx"])]
        elif not array.ndim:
            ego_start[name] = array

    metadata = {
        "probe_id": str(probe_id),
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
            "reference_file": probe.get("reference_file"),
            "replay_start_metadata": replay_start_metadata,
            "replay_start_fingerprint": branch_records[0]["replay_start_fingerprint"],
            "replay_start_observation_fingerprint": branch_records[0]["replay_start_observation_fingerprint"],
        },
        "scene": {
            "map_path": str(map_path),
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
            "agents_per_batch": [int(agent_count) * NUM_ACTIONS],
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
            "family": "ppo",
            "checkpoint": models.audit["checkpoint"],
            "checkpoint_sha256": models.audit["checkpoint_sha256"],
            "gamma": float(models.gamma),
            "gae_lambda": float(models.train_config["gae_lambda"]),
            "vtrace_rho_clip": float(models.train_config["vtrace_rho_clip"]),
            "vtrace_c_clip": float(models.train_config["vtrace_c_clip"]),
            "bptt_horizon": PPO_SEGMENT_ROWS,
            "target_definition": "V(s0) + native GAE advantage at PPO learner slot 0",
            "target_segment_transitions": PPO_SEGMENT_TRANSITIONS,
            "target_importance_ratio": 1.0,
            "target_has_entropy_term": False,
            "normalize_rewards": bool(models.train_config.get("normalize_rewards", False)),
            "use_value_bootstrapping": bool(models.train_config.get("use_value_bootstrapping", True)),
        },
        "branches": branches,
    }
    arrays = {
        "rewards": rewards,
        "training_rewards": training_rewards,
        "log_probs": log_probs,
        "valid": valid,
        "terminals": terminals,
        "truncations": truncations,
        "ego_action_idx": ego_action_idx,
        "values": values,
        "value_valid": value_valid,
        "goal_events": goal_events,
        "action_idx": np.asarray([record["task"]["action_idx"] for record in branch_records], dtype=np.int32),
        "repeat_idx": np.asarray([record["task"]["repeat_idx"] for record in branch_records], dtype=np.int32),
        "continuation_seed": np.asarray([record["task"]["continuation_seed"] for record in branch_records], dtype=np.int64),
        "task_id": np.asarray([record["task"]["task_id"] for record in branch_records], dtype="U160"),
        "initial_value": np.asarray([record["state_values"][0] for record in branch_records], dtype=np.float32),
        "initial_probability": np.asarray([
            record["initial_probabilities"][record["task"]["action_idx"]] for record in branch_records
        ], dtype=np.float32),
        **ppo_arrays,
    }
    arrays["initial_ego_observation"] = branch_records[0]["initial_ego_observation"]
    for name, values_ in safety_arrays.items():
        arrays[f"event_{name}"] = values_
    for name, array_name in zip(metric_names, metric_array_names):
        arrays[array_name] = np.asarray([record["metrics_mean"][name] for record in branch_records], dtype=np.float64)
    for name, value in reference_fields.items():
        arrays[f"scene_start__{name}"] = value
    return metadata, arrays


def _source_hashes(development_dir, repo_root):
    paths = [
        Path(__file__),
        development_dir / "run_probes.py",
        development_dir / "collect_binned_probes.py",
        development_dir / "ppo_models.py",
        development_dir / "probe_env.py",
        development_dir / "initial_geometry.py",
        development_dir / "native_probe" / "build_private_binding.py",
        development_dir / "native_probe" / "probe_export.h",
        development_dir / "native_probe" / "probe_horizon.h",
        repo_root / "pufferlib" / "pufferl.py",
        repo_root / "pufferlib" / "extensions" / "pufferlib.cpp",
        repo_root / "pufferlib" / "ocean" / "drive" / "drive.h",
        repo_root / "pufferlib" / "ocean" / "drive" / "binding.c",
        repo_root / "pufferlib" / "ocean" / "drive" / "drive.py",
    ]
    return source_hashes(paths)


def validate_args(args):
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("--worker-index must be in [0, workers)")


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
    source_records, geometry_manifest, geometry_settings, checkpoint_path, env_config = binned._load_geometry_source(
        source_model_dir, repo_root
    )
    model_name = output_model_dir.name
    if model_name.lower() != "ppo":
        raise ValueError("PPO probe output directory must be named ppo")
    selection_path = records_directory(output_model_dir) / "selection.json"
    selection = binned._load_selection(
        selection_path, source_records=source_records,
        geometry_manifest=geometry_manifest, model_name=model_name,
    )
    seed = int(selection.get("seed", geometry_settings.get("seed", 20261122)))
    if seed < 0:
        raise ValueError("selection seed must be non-negative")
    selection_sha = sha256_file(selection_path)
    records_dir = records_directory(output_model_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    assigned_states = sorted(
        binned._worker_assignment(selection["states"], args.worker_index, args.workers),
        key=lambda state: (
            str(state["source_shard"]), int(state["frame_idx"]), int(state["slot"]), str(state["state_id"])
        ),
    )

    with binned._worker_lock(records_dir, args.worker_index, args.workers):
        binned._worker_progress(
            records_dir, args.worker_index, args.workers,
            status="initializing", worker_id=args.worker_index, worker_count=args.workers,
            assigned_state_count=len(assigned_states), completed_state_count=0,
        )
        from . import probe_env
        from .ppo_models import FrozenPPO

        private_binding = probe_env.ensure_private_binding()
        native_fingerprint = _native_build_fingerprint(
            private_binding, probe_env.native_build_fingerprint()
        )
        source_file_hashes = _source_hashes(development_dir, repo_root)
        map_records = geometry_settings.get("maps", [])
        map_hashes = {record["path"]: record["sha256"] for record in map_records}
        checkpoint_config_path = checkpoint_path.with_name("config.yaml")
        if not checkpoint_config_path.is_file():
            raise FileNotFoundError(f"PPO checkpoint config not found: {checkpoint_config_path}")
        checkpoint_config = yaml.safe_load(checkpoint_config_path.read_text(encoding="utf-8"))
        train_config = checkpoint_config.get("train", {})
        if int(train_config.get("bptt_horizon", -1)) != PPO_SEGMENT_ROWS:
            raise ValueError("PPO checkpoint bptt_horizon must equal the 128-slot target segment")
        run_settings = {
            "source_model_dir": str(source_model_dir),
            "source_geometry_manifest_signature": geometry_manifest["signature"],
            "selection_file_sha256": selection_sha,
            "selection_seed": seed,
            "worker_count": int(args.workers),
            "model_family": "ppo",
            "action_count": NUM_ACTIONS,
            "repeats": REPEATS,
            "continuation_horizon": PROBE_HORIZON,
            "target_segment_rows": PPO_SEGMENT_ROWS,
            "target_segment_transitions": PPO_SEGMENT_TRANSITIONS,
            "branch_batch_size": NUM_ACTIONS,
            "device": str(args.device),
            "continuation_seed_scope": "shared_across_forced_actions_for_each_state",
            "target_importance_ratio": 1.0,
            "prefix_replay": "compact_native_joint_action_replay_with_endpoint_fingerprint",
            "eval_metric_collection": {
                "compute_eval_metrics_during_prefix": False,
                "compute_eval_metrics_after_prefix": True,
                "scope": "probing_agent_forced_action_to_branch_end",
            },
            "action_table_normalized": probe_env._jerk_action_table().tolist(),
            "target_alignment": (
                "128 learner slots; row 0 is initial state with reward 0; action reward/done shift to destination slot; "
                "first 127 transitions feed V(s0)+GAE"
            ),
        }
        manifest = open_or_create_manifest(records_dir, {
            "model_family": "ppo",
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "config_path": str(checkpoint_config_path),
                "config_sha256": sha256_file(checkpoint_config_path),
                "source_audit": geometry_settings.get("source_checkpoint_audit", {}),
            },
            "resolved_config": {
                "source_env": env_config,
                "source_env_config": geometry_settings.get("source_env_config"),
                "source_env_resolved_config": geometry_settings.get("source_env_resolved_config"),
                "gamma": float(train_config["gamma"]),
                "gae_lambda": float(train_config["gae_lambda"]),
                "bptt_horizon": PPO_SEGMENT_ROWS,
                "vtrace_rho_clip": float(train_config["vtrace_rho_clip"]),
                "vtrace_c_clip": float(train_config["vtrace_c_clip"]),
                "use_value_bootstrapping": bool(train_config.get("use_value_bootstrapping", True)),
                "normalize_rewards": bool(train_config.get("normalize_rewards", False)),
                "policy": checkpoint_config.get("policy"),
                "train": train_config,
            },
            "experiment_env_overrides": {
                "compute_eval_metrics": "disabled for prefix replay, enabled at forced-action boundary",
                "probe_horizon": PROBE_HORIZON,
                "target_segment_rows": PPO_SEGMENT_ROWS,
                "source_geometry_experiment_overrides": geometry_settings.get("experiment_env_overrides", {}),
            },
            "map_hashes": map_hashes,
            "native_build_fingerprint": native_fingerprint,
            "source_hashes": source_file_hashes,
            "run_settings": run_settings,
        })
        run_signature = manifest["signature"]
        binned._worker_progress(
            records_dir, args.worker_index, args.workers,
            status="running", run_signature=run_signature,
            assigned_state_count=len(assigned_states), completed_state_count=0,
        )

        completed_states = 0
        completed_branches = 0
        models = None
        try:
            groups = defaultdict(list)
            for state in assigned_states:
                groups[str(state["source_shard"])].append(state)
            for shard_name in sorted(groups):
                states = groups[shard_name]
                shard_path, sidecar_path, episode_metadata, arrays, _, reference = binned._load_source_episode(
                    source_records, geometry_manifest["signature"], states[0]
                )
                map_path = Path(episode_metadata["map_path"]).expanduser().resolve()
                map_hash = sha256_file(map_path)
                if map_hash != episode_metadata.get("map_sha256"):
                    raise ValueError(f"source map hash differs from geometry sidecar: {map_path}")
                if map_hashes.get(str(map_path)) != map_hash:
                    raise ValueError(f"source map is not covered by geometry manifest: {map_path}")
                if int(episode_metadata["map_index"]) != int(states[0]["map_index"]):
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
                        completed_states += 1
                        completed_branches += NUM_ACTIONS
                        continue
                    incomplete_states.append(state)
                if not incomplete_states:
                    continue

                layout_seed = int(episode_metadata["layout_seed"])
                template = probe_env.create_template(env_config, seed=layout_seed)
                try:
                    agent_count = int(episode_metadata["agent_count"])
                    if template.map_agent_counts.get(str(map_path)) != agent_count:
                        raise ValueError("recreated PPO layout agent count differs from geometry reference")
                    if models is None:
                        models = FrozenPPO(checkpoint_path, template.model_env, device=args.device)
                        expected_hashes = geometry_settings.get("source_checkpoint_audit", {}).get("model_hashes")
                        if expected_hashes and models.initial_hashes != expected_hashes:
                            raise ValueError("loaded PPO model hashes differ from source geometry audit")
                        if models.audit["checkpoint_sha256"] != geometry_settings["source_checkpoint_sha256"]:
                            raise ValueError("loaded PPO checkpoint differs from source geometry manifest")
                        if not np.allclose(models.action_table, probe_env._jerk_action_table(), rtol=0.0, atol=1e-7):
                            raise ValueError("PPO checkpoint action table differs from native jerk action order")
                        if float(models.gamma) != float(train_config["gamma"]):
                            raise ValueError("loaded PPO gamma differs from saved training config")
                    for state in incomplete_states:
                        source_row = binned._source_row_for_state(state, episode_metadata, arrays)
                        frame_idx = int(state["frame_idx"])
                        ego_idx = int(state["slot"])
                        probe_id = str(state["state_id"])
                        probe = {
                            "frame_idx": frame_idx,
                            "ego_idx": ego_idx,
                            "agent_id": int(state["agent_id"]),
                            "slot": ego_idx,
                            "state_id": probe_id,
                            "assigned_bin": binned._assigned_bin(state),
                            "reference_file": str(Path(state["source_shard"]).as_posix()),
                        }

                        def report_progress(frame_count, live_count):
                            if frame_count % 512 == 0:
                                print(json.dumps({
                                    "worker": args.worker_index,
                                    "state_id": probe_id,
                                    "branch_frames": frame_count,
                                    "live_branches": live_count,
                                }, sort_keys=True), flush=True)

                        branch_records = _run_branch_batch(
                            probe_env.SceneBatch, template, str(map_path), agent_count,
                            int(episode_metadata["scene_seed"]), reference, probe,
                            state_tasks[probe_id], models, progress_callback=report_progress,
                        )
                        binned._verify_replayed_state(branch_records, state, source_row)
                        probe_metadata, output_arrays = _assemble_probe_bundle(
                            probe_id, int(state["episode_index"]), str(map_path), map_hash,
                            int(episode_metadata["scene_seed"]), reference, probe, agent_count,
                            branch_records, models, num_goals=int(template.probe_config["num_goals"]),
                        )
                        probe_metadata.update({
                            "format_version": PPO_COLLECTOR_FORMAT_VERSION,
                            "state_id": probe_id,
                            "assigned_bin": binned._assigned_bin(state),
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
                                "target_segment_rows": PPO_SEGMENT_ROWS,
                                "target_segment_transitions": PPO_SEGMENT_TRANSITIONS,
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
                        probe_metadata["selection_file"] = str(selection_path.relative_to(records_dir))
                        _, data_path = probe_paths(records_dir, probe_id)
                        save_probe(records_dir, probe_id, run_signature, probe_metadata, output_arrays)
                        completed_states += 1
                        completed_branches += NUM_ACTIONS
                        binned._worker_progress(
                            records_dir, args.worker_index, args.workers,
                            status="running", run_signature=run_signature,
                            assigned_state_count=len(assigned_states),
                            completed_state_count=completed_states,
                            completed_branch_count=completed_branches,
                            current_state_id=probe_id,
                        )
                        print(json.dumps({
                            "worker": args.worker_index,
                            "state_id": probe_id,
                            "completed_states": completed_states,
                            "assigned_states": len(assigned_states),
                            "branches": NUM_ACTIONS,
                            "data_file": data_path.name,
                        }, sort_keys=True), flush=True)
                        models.verify_frozen()
                finally:
                    template.close()

            summary = binned._update_collection_summary(records_dir, selection, run_signature)
            binned._worker_progress(
                records_dir, args.worker_index, args.workers,
                status="complete", run_signature=run_signature,
                assigned_state_count=len(assigned_states),
                completed_state_count=completed_states,
                completed_branch_count=completed_branches,
                overall_completed_state_count=summary["completed_state_count"],
                overall_expected_state_count=summary["selection_state_count"],
                overall_complete=summary["complete"],
            )
            return summary
        except Exception as error:
            binned._worker_progress(
                records_dir, args.worker_index, args.workers,
                status="failed", run_signature=run_signature,
                error=f"{type(error).__name__}: {error}",
                completed_state_count=completed_states,
                completed_branch_count=completed_branches,
            )
            raise
        finally:
            if models is not None:
                models.verify_frozen()


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    binned.validate_args(args)
    return run_collection(args)


if __name__ == "__main__":
    main()
