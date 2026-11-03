"""Frozen-policy Q calibration rollouts; all output stays in this project."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import yaml

from project.soft_actor_critic_q_calibration.diagnostic import (
    get_diagnostics, install_private_binding,
)
from project.soft_actor_critic_q_calibration.returns import compute_returns


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
SOURCE = ROOT / "project/soft_actor_critic_target_entropy_scheduling/runs/round2_tes_sac_20261029T044847.014369240_539543/trial_98/learner"
FLOAT_FIELDS = (
    "q1", "q2", "target_q1", "target_q2", "entropy", "graph_distance_m",
    "euclidean_distance_m", "observed_goal_distance_m", "speed_mps",
    "goal_reward_coeff", "collision_coeff", "offroad_coeff", "comfort_coeff",
    "goal_radius_m", "goal_speed_mps", "goal_mismatch", "goal_index", "goal_count",
    "graph_status", "action", "scene_id", "episode_id", "agent_slot", "step",
    "scene_timestep", "map_id", "goal_hit_next", "native_agent_id", "slot_idx",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_models(env, config, checkpoint, device, compile_models):
    from pufferlib.ocean.torch_sac import DriveSAC, DriveSACCritic
    actor = DriveSAC(env, **config["policy"]).to(device).eval()
    critic = DriveSACCritic(env, **config["policy"]).to(device).eval()
    target = DriveSACCritic(env, **config["policy"]).to(device).eval()
    actor.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    target.load_state_dict(checkpoint["target_critic_state_dict"], strict=True)
    for model in (actor, critic, target):
        model.requires_grad_(False)
    actor_call, critic_call, target_call = actor, critic, target
    if compile_models:
        actor_call = torch.compile(actor, mode="default")
        critic_call = torch.compile(critic, mode="default")
        target_call = torch.compile(target, mode="default")
    return actor, actor_call, critic_call, target_call


@torch.inference_mode()
def state_predictions(observations, actor, critic, target, device, alpha, amp):
    tensor = torch.as_tensor(observations, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        logits = actor(tensor)
        q1, q2 = critic(tensor)
        target_q1, target_q2 = target(tensor)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probabilities = log_probs.exp()
    if not torch.isfinite(log_probs).all():
        raise ValueError("Nonfinite frozen-policy log probabilities")
    soft_values = (probabilities * (torch.minimum(target_q1.float(), target_q2.float()) - alpha * log_probs)).sum(-1)
    return log_probs, probabilities, q1.float(), q2.float(), target_q1.float(), target_q2.float(), soft_values


@torch.inference_mode()
def endpoint_values(observations, actor, target, device, alpha, amp):
    tensor = torch.as_tensor(observations, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        logits = actor(tensor)
        q1, q2 = target(tensor)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return (log_probs.exp() * (torch.minimum(q1.float(), q2.float()) - alpha * log_probs)).sum(-1).cpu().numpy()


def record_metadata(records, sample_idx, metadata, observations, env, episode_ids, actions, predictions):
    row_count = env.num_agents
    goal_indices = metadata["current_goal_idx"].astype(np.int64)
    valid_goal = (goal_indices >= 0) & (goal_indices < env.num_goals) & (goal_indices < metadata["goal_count"])
    safe_indices = np.clip(goal_indices, 0, env.num_goals - 1)
    goal_offset = env.ego_features + env.num_reward_coefs
    goal_observations = observations[:, goal_offset:goal_offset + env.goal_dim].reshape(row_count, env.num_goals, 3)
    selected_goal = goal_observations[np.arange(row_count), safe_indices]
    observed_distance = np.linalg.norm(selected_goal[:, :2], axis=-1) * env.obs_norm_goal_offset_m
    observed_distance[~valid_goal] = np.nan
    native_distance = metadata["goal_distance_xy_meters"]
    goal_mismatch = (~valid_goal) | (~np.isfinite(native_distance)) | (np.abs(observed_distance - native_distance) > 0.02)
    coefficients = metadata["reward_coefficients"]
    q1, q2, tq1, tq2, entropy = predictions
    values = {
        "q1": q1, "q2": q2, "target_q1": tq1, "target_q2": tq2,
        "entropy": entropy, "graph_distance_m": metadata["graph_distance_meters"],
        "euclidean_distance_m": native_distance, "observed_goal_distance_m": observed_distance,
        "speed_mps": metadata["speed_mps"], "goal_reward_coeff": env.reward_goal,
        "collision_coeff": coefficients[:, 2], "offroad_coeff": coefficients[:, 3],
        "comfort_coeff": coefficients[:, 4], "goal_radius_m": coefficients[:, 0],
        "goal_speed_mps": coefficients[:, 1], "goal_mismatch": goal_mismatch,
        "goal_index": goal_indices, "goal_count": metadata["goal_count"],
        "graph_status": metadata["graph_distance_status"], "action": actions,
        "scene_id": metadata["scene_idx"], "episode_id": episode_ids,
        "agent_slot": np.arange(row_count), "scene_timestep": metadata["timestep"],
        "native_agent_id": metadata["agent_id"], "slot_idx": metadata["slot_idx"],
        "map_id": np.asarray(env.map_ids)[np.maximum(metadata["scene_idx"], 0)],
    }
    for name, value in values.items():
        records[name][sample_idx] = value


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        raise ValueError("All analysis output must remain inside the calibration folder")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") not in {"6", "7", "6,7"}:
        raise ValueError("CUDA_VISIBLE_DEVICES must restrict analysis to physical GPU 6 and/or 7")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.action_seed)
    np.random.seed(args.env_seed)
    config = yaml.safe_load((SOURCE / "config.yaml").read_text())
    checkpoint_path = SOURCE / "trainer_state.pt"
    # Trusted, locally produced learner checkpoint includes NumPy RNG metadata.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    alpha = float(checkpoint["log_alpha"].exp())
    gamma = float(checkpoint["config"]["gamma"])
    if not checkpoint["config"].get("critic_entropy_in_target", True):
        raise ValueError("This diagnostic assumes a soft Q critic")
    install_private_binding()
    from pufferlib.ocean.drive.drive import Drive
    env_config = dict(config["env"])
    env_config["capture_final_observations"] = True
    # Worker count is reduced; each Drive instance preserves the saved native scene settings.
    if args.num_agents is not None:
        env_config["num_agents"] = args.num_agents
    env_config["seed"] = args.env_seed
    env = Drive(**env_config)
    audit = {
        "run_id": output.name, "status": "running", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_hash(checkpoint_path), "source_config_sha256": file_hash(SOURCE / "config.yaml"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "training_git_commit": config.get("git", {}).get("commit_hash"),
        "checkpoint_epoch": checkpoint["epoch"], "agent_steps": checkpoint["agent_steps"],
        "alpha": alpha, "gamma": gamma, "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env_seed": args.env_seed, "action_seed": args.action_seed, "steps": args.steps,
        "record_stride": args.record_stride, "env_config": env_config,
        "native_scene_count": env.num_envs, "map_ids": np.asarray(env.map_ids).tolist(),
        "observation_size": env.num_obs, "action_selection": "sample",
        "amp_bfloat16": bool(config["train"].get("amp", False) and device.type == "cuda"),
        "compiled": args.compile, "distance_definition": "raw current-lane to active-goal-lane precomputed graph metric; no within-lane correction",
        "analysis_source_sha256": {str(path.relative_to(PROJECT)): file_hash(path)
                                   for path in (PROJECT / "collect.py", PROJECT / "returns.py",
                                                PROJECT / "diagnostic.py", PROJECT / "native/diagnostic_export.h")},
        "initialization_cluster": "run_id,scene_id,episode_id; independent worker seeds, no repeated-state branching",
    }
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config | {"env": env_config}, sort_keys=False))
    write_json(output / "run.json", audit)
    started = time.monotonic()
    actor_model, actor, critic, target = frozen_models(env, config, checkpoint, device, args.compile)
    frozen_before = [model_hash(model) for model in (actor_model, critic, target)]
    observations, _ = env.reset(seed=args.env_seed)
    shape = (args.steps, env.num_agents)
    rewards = np.full(shape, np.nan, np.float32)
    log_probs_history = np.full(shape, np.nan, np.float32)
    next_values = np.full(shape, np.nan, np.float32)
    valid = np.zeros(shape, bool)
    terminals = np.zeros(shape, bool)
    truncations = np.zeros(shape, bool)
    sample_count = (args.steps + args.record_stride - 1) // args.record_stride
    records = {key: np.full((sample_count, env.num_agents), np.nan, np.float32) for key in FLOAT_FIELDS}
    records["episode_seed"] = np.zeros((sample_count, env.num_agents), np.uint64)
    reward_coefficients = np.full((sample_count, env.num_agents, env.num_reward_coefs), np.nan, np.float32)
    episode_ids = np.zeros(env.num_agents, np.int32)
    previous_seeds = None
    previous_times = None
    sample_steps = np.arange(0, args.steps, args.record_stride)
    for step_idx in range(args.steps):
        metadata = get_diagnostics(env)
        seeds = metadata["episode_seed"]
        times = metadata["timestep"]
        if previous_seeds is not None:
            episode_ids += ((seeds != previous_seeds) | (times < previous_times)).astype(np.int32)
        previous_seeds, previous_times = seeds.copy(), times.copy()
        predictions = state_predictions(observations, actor, critic, target, device, alpha, audit["amp_bfloat16"])
        log_probs, probabilities, q1, q2, tq1, tq2, soft_values = predictions
        if step_idx:
            ordinary = valid[step_idx - 1] & ~terminals[step_idx - 1] & ~truncations[step_idx - 1]
            next_values[step_idx - 1, ordinary] = soft_values.cpu().numpy()[ordinary]
        actions_tensor = torch.multinomial(probabilities, 1).squeeze(-1)
        actions = actions_tensor.cpu().numpy()
        selected_log_probs = log_probs.gather(1, actions_tensor[:, None]).squeeze(-1).cpu().numpy()
        entropy = -(probabilities * log_probs).sum(-1).cpu().numpy()
        continuous_actions = actor_model.discrete_actions_to_continuous(actions_tensor).cpu().numpy()
        sample_idx = step_idx // args.record_stride
        if step_idx % args.record_stride == 0:
            records["episode_seed"][sample_idx] = seeds
            reward_coefficients[sample_idx] = metadata["reward_coefficients"]
            selected_q = [values.gather(1, actions_tensor[:, None]).squeeze(-1).cpu().numpy() for values in (q1, q2, tq1, tq2)]
            record_metadata(records, sample_idx, metadata, observations, env, episode_ids, actions, (*selected_q, entropy))
            records["step"][sample_idx] = step_idx
        observations, step_rewards, step_terminals, step_truncations, infos = env.step(continuous_actions)
        # Returned masks describe the just-completed transition, as in the SAC collector.
        valid[step_idx] = np.asarray(env.masks, dtype=bool)
        rewards[step_idx] = step_rewards
        log_probs_history[step_idx] = selected_log_probs
        terminals[step_idx] = np.asarray(step_terminals, bool) & valid[step_idx]
        truncations[step_idx] = np.asarray(step_truncations, bool) & valid[step_idx] & ~terminals[step_idx]
        next_values[step_idx, terminals[step_idx]] = 0
        truncated_rows = valid[step_idx] & truncations[step_idx] & ~terminals[step_idx]
        if truncated_rows.any():
            if not np.all(env.final_observation_valid[truncated_rows]):
                raise ValueError("Missing pre-reset successor for a valid truncated transition")
            next_values[step_idx, truncated_rows] = endpoint_values(env.final_observations[truncated_rows], actor, target, device, alpha, audit["amp_bfloat16"])
        if step_idx % args.record_stride == 0:
            after = get_diagnostics(env)
            same_episode = after["episode_seed"] == seeds
            records["goal_hit_next"][sample_idx] = np.where(same_episode, after["reached_goal_metric"], np.nan)
        if (step_idx + 1) % args.log_steps == 0 or step_idx + 1 == args.steps:
            elapsed = time.monotonic() - started
            progress = {"step": step_idx + 1, "total_steps": args.steps, "elapsed_seconds": elapsed,
                        "valid_rows": int(valid[:step_idx + 1].sum()), "rows_per_second": float(valid[:step_idx + 1].sum() / elapsed)}
            write_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    last_rows = valid[-1] & ~terminals[-1] & ~truncations[-1]
    if last_rows.any():
        next_values[-1, last_rows] = endpoint_values(observations[last_rows], actor, target, device, alpha, audit["amp_bfloat16"])
    frozen_after = [model_hash(model) for model in (actor_model, critic, target)]
    if frozen_before != frozen_after:
        raise AssertionError("Frozen actor or critic state changed during diagnostic")
    env.close()
    estimates = compute_returns(rewards, log_probs_history, valid, terminals, truncations, next_values, gamma, alpha)
    selected_valid = valid[sample_steps]
    output_arrays = {key: value[selected_valid] for key, value in records.items()}
    output_arrays["reward_coefficients"] = reward_coefficients[selected_valid]
    mapped = {"g_reward": estimates.reward_partial, "g_soft": estimates.soft_partial,
              "entropy_return": estimates.entropy_contribution, "censored": estimates.censored,
              "horizon": estimates.observed_horizon, "tail_discount": estimates.tail_discount,
              "bootstrap_tail": estimates.bootstrap_tail, "g_soft_bootstrap": estimates.soft_bootstrapped,
              "truncation_endpoint": estimates.truncation_endpoint,
              "reward": rewards, "log_prob": log_probs_history, "terminal": terminals, "truncated": truncations}
    output_arrays.update({key: value[sample_steps][selected_valid] for key, value in mapped.items()})
    np.savez_compressed(output / "transitions.npz", **output_arrays)
    audit.update(status="completed", elapsed_seconds=time.monotonic() - started,
                 valid_transition_count=int(valid.sum()), recorded_rows=int(selected_valid.sum()),
                 terminal_transition_count=int(terminals.sum()),
                 truncation_transition_count=int(truncations.sum()),
                 completed_return_fraction=float(np.mean(~output_arrays["censored"])),
                 goal_mismatch_fraction=float(np.mean(output_arrays["goal_mismatch"])),
                 graph_available_fraction=float(np.mean(np.isfinite(output_arrays["graph_distance_m"]))),
                 actor_unchanged=True, critics_unchanged=True, model_hashes=frozen_after,
                 action_count=actor_model.action_count)
    write_json(output / "run.json", audit)
    print(json.dumps({key: audit[key] for key in ("status", "elapsed_seconds", "valid_transition_count", "recorded_rows", "completed_return_fraction")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2560)
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--env-seed", type=int, required=True)
    parser.add_argument("--action-seed", type=int, required=True)
    parser.add_argument("--record-stride", type=int, default=8)
    parser.add_argument("--log-steps", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.steps, arguments.record_stride, arguments.log_steps, arguments.cpu_threads) <= 0:
        parser.error("steps, stride, log steps, and CPU threads must be positive")
    run(arguments)
