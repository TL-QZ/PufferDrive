"""Native checks for extending verified gigaflow probe scenes."""

from pathlib import Path

import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.development.probe_env import (
    SceneBatch,
    create_template,
    ensure_private_binding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MAP_PATH = PROJECT_ROOT / "pufferlib" / "resources" / "drive" / "binaries" / "carla" / "opendrive__Town03.bin"
SCENE_STEPS = 2560
PROBE_STEPS = 2560


def _env_config(scenario_length=SCENE_STEPS):
    return {
        "render_mode": "headless",
        "report_interval": 1,
        "dt": 0.3,
        "scenario_length": scenario_length,
        "resample_frequency": 10000,
        "resample_replay_to_dt": False,
        "obs_range_traffic_control_m": 10000.0,
        "num_maps": 1,
        "num_agents": 4,
        "action_type": "continuous",
        "dynamics_model": "jerk",
        "simulation_mode": "gigaflow",
        "termination_mode": 0,
        "inactive_agent_threshold": 0.4,
        "terminate_on_goal": False,
        "seed": 19,
        "init_step": 0,
        "init_step_spread": False,
        "map_dir": str(MAP_PATH),
        "goal_regen_mode": "finite",
        "goal_source": "map",
        "num_goals": 3,
        "min_goal_spacing": 20.0,
        "max_goal_spacing": 60.0,
        "reward_conditioning": True,
        "reward_randomization": True,
        "compute_eval_metrics": False,
        "eval_mode": 0,
        "min_agents_per_env": 4,
        "max_agents_per_env": 4,
    }


def _scene_batch(template, scene_seed, copies=1):
    return SceneBatch.from_template(
        template,
        template.template_map_path,
        template.template_agent_count,
        copies=copies,
        scene_seed=scene_seed,
    )


def _periodic_suffix(states, new_size):
    state_count = len(states)
    for period in range(1, state_count // 2 + 1):
        if all(states[index] == states[index - period] for index in range(period, state_count)):
            return np.asarray([states[index % period] for index in range(state_count, new_size)])
    raise AssertionError("Test traffic signal does not contain two observed cycles")


@pytest.mark.skipif(not MAP_PATH.is_file(), reason="CARLA test map is not available")
def test_verified_late_prefix_gets_2560_fresh_steps_and_keeps_prefix_state():
    ensure_private_binding()
    template = create_template(_env_config(), seed=101)
    reference = None
    batch = None
    scene_seed = 7301
    prefix_frame = SCENE_STEPS - 1
    prefix = np.full((prefix_frame, template.template_agent_count), 7, dtype=np.int32)
    try:
        reference = _scene_batch(template, scene_seed)
        expected_fingerprint = reference.replay_prefix(prefix, prefix_frame)[0]
        batch = _scene_batch(template, scene_seed)
        batch.replay_prefix(prefix, prefix_frame, expected_fingerprint=expected_fingerprint, compact=True)

        observations_before = batch.observations.copy()
        metadata_before = batch.metadata()
        batch.start_probe_horizon(PROBE_STEPS)
        metadata_after = batch.metadata()
        new_cutoff = prefix_frame + PROBE_STEPS

        assert np.array_equal(batch.observations, observations_before)
        for name in (
            "pose_xyz", "goal_xyz", "heading_rad", "speed_mps", "reward_coefficients",
            "timestep", "episode_seed", "current_goal_idx", "goal_count", "num_goals_reached",
            "traffic_control_type", "traffic_control_pose",
        ):
            np.testing.assert_array_equal(metadata_after[name], metadata_before[name])

        from pufferlib.ocean.drive import binding

        light_slots = np.flatnonzero(
            metadata_before["traffic_control_type"][0] == binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT
        )
        assert light_slots.size > 0
        for control_idx in light_slots:
            original_state_size = int(metadata_before["traffic_control_state_size"][0, control_idx])
            assert original_state_size == SCENE_STEPS
            assert int(metadata_after["traffic_control_state_size"][0, control_idx]) == new_cutoff
            original_states = metadata_before["traffic_control_state"][0, control_idx, :original_state_size]
            extended_states = metadata_after["traffic_control_state"][0, control_idx, :new_cutoff]
            np.testing.assert_array_equal(extended_states[:original_state_size], original_states)
            np.testing.assert_array_equal(
                extended_states[original_state_size:],
                _periodic_suffix(original_states, new_cutoff),
            )

        metrics_baseline = batch.begin_metrics(ego_idx=0)
        batch.actions[:] = batch.action_table[7]
        for _ in range(PROBE_STEPS):
            batch.binding.vec_step(batch.c_envs)

        final_metadata = batch.metadata(include_traffic=False)
        assert final_metadata["timestep"].tolist() == [new_cutoff]
        assert final_metadata["scene_done"].tolist() == [True]
        assert np.all(batch.truncations)
        assert np.all(batch.final_observation_valid)

        traffic_offset = (
            template.ego_features
            + template.num_reward_coefs
            + template.goal_dim
            + template.obs_slots_partners_n * template.partner_features
            + template.obs_slots_lane_kept * template.lane_features
            + template.obs_slots_boundary_kept * template.boundary_features
        )
        terminal_observation = batch.final_observations[0]
        traffic_features = template.traffic_control_features
        observed_light_states = []
        for agent_observation in terminal_observation:
            for control_slot in range(template.obs_slots_traffic_controls_n):
                slot_offset = traffic_offset + control_slot * traffic_features
                control_type = int(agent_observation[slot_offset + 5])
                if control_type == binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT:
                    observed_light_states.append(int(agent_observation[slot_offset + 6]))
        assert observed_light_states
        assert set(observed_light_states) == {binding.TRAFFIC_CONTROL_STATE_OFF}

        interval_metrics = batch.metrics(metrics_baseline, np.zeros((1, 4), dtype=np.bool_))[0]
        assert interval_metrics["episode_timestep"] == PROBE_STEPS
        assert all(np.isfinite(value) for value in interval_metrics.values())
    finally:
        if batch is not None:
            batch.close()
        if reference is not None:
            reference.close()
        template.close()


@pytest.mark.skipif(not MAP_PATH.is_file(), reason="CARLA test map is not available")
def test_probe_metric_enablement_preserves_the_next_transition_and_checks_prefix_hash():
    ensure_private_binding()
    template = create_template(_env_config(scenario_length=32), seed=202)
    reference = None
    batch = None
    scene_seed = 8402
    prefix = np.full((1, template.template_agent_count), 7, dtype=np.int32)
    try:
        reference = _scene_batch(template, scene_seed)
        expected_fingerprint = reference.replay_prefix(prefix, 1)[0]
        batch = _scene_batch(template, scene_seed)
        batch.replay_prefix(prefix, 1, expected_fingerprint=expected_fingerprint, compact=True)

        action_before = batch.actions.copy()
        batch.actions[0, 0, 0] = np.float32(0.25)
        with pytest.raises(AssertionError, match="fingerprint changed"):
            batch.start_probe_horizon(20)
        batch.actions[:] = action_before

        batch.start_probe_horizon(20)
        metrics_baseline = batch.begin_metrics(ego_idx=0)
        private_prefix_observation = batch.observations.copy()
        reference_prefix_observation = reference.observations.copy()
        normalized_action = np.repeat(
            batch.action_table[7][None, :], template.template_agent_count, axis=0
        )
        private_step = batch.step(normalized_action[None, ...])
        reference_step = reference.step(normalized_action[None, ...])
        np.testing.assert_array_equal(private_prefix_observation[0], reference_prefix_observation[0])
        np.testing.assert_array_equal(private_step["observations"], reference_step["observations"])
        np.testing.assert_array_equal(private_step["rewards"], reference_step["rewards"])
        np.testing.assert_array_equal(private_step["terminals"], reference_step["terminals"])
        np.testing.assert_array_equal(private_step["truncations"], reference_step["truncations"])
        private_metadata = batch.metadata(include_traffic=False)
        reference_metadata = reference.metadata(include_traffic=False)
        for name in ("pose_xyz", "goal_xyz", "heading_rad", "speed_mps", "current_goal_idx", "goal_count"):
            np.testing.assert_array_equal(private_metadata[name], reference_metadata[name])
        flags = np.asarray(
            [[private_step["events"][name][0, 0] for name in ("collision", "at_fault_collision", "offroad", "red_light")]],
            dtype=np.bool_,
        )
        interval_metrics = batch.metrics(metrics_baseline, flags)[0]
        assert interval_metrics["episode_timestep"] == 1
        assert all(np.isfinite(value) for value in interval_metrics.values())
    finally:
        if batch is not None:
            batch.close()
        if reference is not None:
            reference.close()
        template.close()
