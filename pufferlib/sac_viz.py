"""Value-free interactive replay encoding for SAC evaluations."""

import numpy as np

import pufferlib.viz
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import compute_effective_road_obs_count
from pufferlib.viz import _obs_scales, _pack_replay_binary


def encode_interactive_replay(scenario, replay):
    road_points = []
    road_lengths = []
    road_types = []
    for elem in scenario.get("road_elements", []) or []:
        if not isinstance(elem, dict):
            continue
        elem_type = int(elem.get("type", 0))
        xs = elem.get("x") or []
        ys = elem.get("y") or []
        if not xs or not ys:
            continue
        if 1 <= elem_type <= 3:
            draw_type = 0
        elif 11 <= elem_type <= 18:
            draw_type = 1
        elif 21 <= elem_type <= 23:
            draw_type = 2
        else:
            continue
        count = min(len(xs), len(ys))
        road_lengths.append(count)
        road_types.append(draw_type)
        for i in range(count):
            road_points.append((float(xs[i]), float(ys[i])))

    traffic_stop_lines = []
    traffic_types = []
    for elem in scenario.get("traffic_elements", []) or []:
        if not isinstance(elem, dict):
            continue
        stop_line = elem.get("stop_line") or [0, 0, 0, 0, 0, 0]
        traffic_stop_lines.append([float(v) for v in stop_line[:6]])
        traffic_types.append(int(elem.get("type", 0)))

    env_cfg = replay["env"]
    scales = _obs_scales(env_cfg)
    lane_count = compute_effective_road_obs_count(env_cfg["obs_slots_lane_n"], env_cfg.get("obs_dropout_lane", 0.0))
    boundary_count = compute_effective_road_obs_count(
        env_cfg["obs_slots_boundary_n"], env_cfg.get("obs_dropout_boundary", 0.0)
    )

    observation_scale = 1.0
    quantized_observations = None
    if replay.get("obs") is not None:
        observations_f32 = np.asarray(replay["obs"], dtype=np.float32)
        if observations_f32.size:
            observation_scale = float(np.max(np.abs(observations_f32))) / 32767.0
        if observation_scale == 0.0:
            observation_scale = 1.0
        quantized_observations = np.round(observations_f32 / observation_scale).astype(np.int16)

    chunks = {
        "road_points": np.asarray(road_points or [(0.0, 0.0)], dtype=np.float32),
        "road_lengths": np.asarray(road_lengths or [0], dtype=np.int32),
        "road_types": np.asarray(road_types or [0], dtype=np.int16),
        "traffic_stop_lines": np.asarray(traffic_stop_lines or [[0, 0, 0, 0, 0, 0]], dtype=np.float32),
        "traffic_types": np.asarray(traffic_types or [0], dtype=np.int16),
        "agent_f32": replay["agent_f32"].astype(np.float32, copy=False),
        "agent_i32": replay["agent_i32"].astype(np.int32, copy=False),
        "metrics_f32": replay["metrics_f32"].astype(np.float32, copy=False),
        "puffer_f32": replay["puffer_f32"].astype(np.float32, copy=False),
        "traffic_i16": replay["traffic_i16"].astype(np.int16, copy=False),
        "raw_action": replay["raw_action"].astype(np.float32, copy=False),
        "clipped_action": replay["clipped_action"].astype(np.float32, copy=False),
        "entropy": replay["entropy"].astype(np.float32, copy=False),
    }
    if quantized_observations is not None:
        chunks["obs"] = quantized_observations
    if replay.get("policy_probs") is not None:
        chunks["policy_probs"] = replay["policy_probs"].astype(np.float32, copy=False)
    if replay.get("policy_mean") is not None:
        chunks["policy_mean"] = replay["policy_mean"].astype(np.float32, copy=False)
        chunks["policy_std"] = replay["policy_std"].astype(np.float32, copy=False)
        chunks["policy_log_prob"] = replay["policy_log_prob"].astype(np.float32, copy=False)
    for pool_name in ("pool_partner", "pool_lane", "pool_boundary", "pool_traffic"):
        if replay.get(pool_name) is not None:
            chunks[pool_name] = replay[pool_name].astype(np.int16, copy=False)

    # Ghost: logged/expert trajectory bbox per active agent, so policy-vs-log divergence is visible
    # (esp. control_sdc_only, 1 active agent). Frame-aligned: frame f = logged pose at timestep init_step + f.
    # Fields: x, y, heading, length, width; width <= 0 marks frames with no valid logged pose.
    agents = scenario.get("agents", []) or []
    active_indices = scenario.get("active_agent_indices", []) or []
    expert_indices = [agent_idx for agent_idx, agent in enumerate(agents) if int(agent.get("mark_as_expert", 0)) == 1]
    frame_count = int(replay["agent_f32"].shape[0])
    active_count = int(replay["raw_action"].shape[1])
    init_step = int(env_cfg.get("init_step", 0))
    ghost = np.zeros((frame_count, max(1, active_count), 5), dtype=np.float32)
    for slot in range(min(active_count, len(active_indices))):
        agent_idx = active_indices[slot]
        if agent_idx < 0 or agent_idx >= len(agents):
            continue
        a = agents[agent_idx]
        lx = np.asarray(a.get("log_trajectory_x") or [], dtype=np.float32)
        ly = np.asarray(a.get("log_trajectory_y") or [], dtype=np.float32)
        lh = np.asarray(a.get("log_heading") or [], dtype=np.float32)
        lv = np.asarray(a.get("log_valid") or [], dtype=np.int32)
        n = min(frame_count, max(0, lx.shape[0] - init_step))
        if n <= 0:
            continue
        window = slice(init_step, init_step + n)
        ghost[:n, slot, 0] = lx[window]
        ghost[:n, slot, 1] = ly[window]
        ghost[:n, slot, 2] = lh[window]
        ghost[:n, slot, 3] = float(a.get("sim_length", 0.0))
        width = float(a.get("sim_width", 0.0))
        ghost[:n, slot, 4] = np.where(lv[window] == 0, 0.0, width) if lv.shape[0] >= init_step + n else width
    chunks["ghost_f32"] = ghost

    metadata = {
        "map_name": scenario.get("map_name", "Unknown"),
        "scenario_id": scenario.get("scenario_id", "Unknown"),
        "expert_indices": expert_indices,
        "total_agents": int(scenario.get("num_total_agents", replay["agent_f32"].shape[1])),
        "eval_overrides": replay.get("eval_overrides") or {},
        "frames": int(replay["agent_f32"].shape[0]),
        "agent_cap": int(replay["agent_f32"].shape[1]),
        "traffic_cap": int(replay["traffic_i16"].shape[1]),
        "active_count": int(replay["raw_action"].shape[1]),
        "obs_dim": int(replay["obs"].shape[2]) if replay.get("obs") is not None else 0,
        "obs_scale": observation_scale,
        "action_type": env_cfg.get("action_type", "continuous"),
        "dynamics_model": env_cfg.get("dynamics_model", "classic"),
        "num_goals": int(env_cfg["num_goals"]),
        "reward_conditioning": bool(env_cfg["reward_conditioning"]),
        "obs_slots_partners_n": int(env_cfg["obs_slots_partners_n"]),
        "ego_dim": int(binding.EGO_FEATURES),
        "reward_coef_count": int(binding.NUM_REWARD_COEFS),
        "partner_features": int(binding.PARTNER_FEATURES),
        "lane_features": int(binding.LANE_FEATURES),
        "boundary_features": int(binding.BOUNDARY_FEATURES),
        "traffic_features": int(binding.TRAFFIC_CONTROL_FEATURES),
        "lane_count": int(lane_count),
        "boundary_count": int(boundary_count),
        "traffic_obs_count": int(env_cfg["obs_slots_traffic_controls_n"]),
        "goal_features": int(binding.GOAL_FEATURES),
        "scales": scales,
        "road_polyline_count": len(road_lengths),
        "traffic_static_count": len(traffic_types),
    }
    return _pack_replay_binary(metadata, chunks)


def save_interactive_replay_zlib(scenario, replay, filename):
    compressed_payload = encode_interactive_replay(scenario, replay)
    with open(filename, "wb") as replay_file:
        replay_file.write(compressed_payload)
    return compressed_payload


def render_interactive_replay_zlib(replay_path, filename):
    """Use the shared viewer, then remove its PPO-only value panel and read."""
    pufferlib.viz.render_interactive_replay_zlib(replay_path, filename)
    with open(filename, "r", encoding="utf-8") as rendered_file:
        html = rendered_file.read()

    # These literals mirror the value row and value read in pufferlib.viz's viewer.
    value_panel = '<div class="item"><span class="name">value</span><span class="num" data-pol="v">-</span></div>'
    value_read = "            refs.polV.textContent = C.value[s].toFixed(3);\n"
    if html.count(value_panel) != 1 or html.count(value_read) != 1:
        raise RuntimeError("Shared replay viewer changed its PPO value markup; update the SAC adapter")
    html = html.replace(value_panel, "").replace(value_read, "")
    with open(filename, "w", encoding="utf-8") as rendered_file:
        rendered_file.write(html)
