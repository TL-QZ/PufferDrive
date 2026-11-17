"""CPU checks for deterministic Drive copies and held-boundary parity."""

from pathlib import Path

import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.probe_env import (
    SceneBatch,
    create_template,
    ensure_private_binding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAP_PATH = PROJECT_ROOT / "pufferlib" / "resources" / "drive" / "binaries" / "carla" / "opendrive__Town03.bin"


def _env_config():
    return {
        "render_mode": "headless",
        "report_interval": 1,
        "dt": 0.3,
        "scenario_length": 3,
        "resample_frequency": 10000,
        "resample_replay_to_dt": False,
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


def _production_env(template, scene_seed):
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.drive import drive as drive_module

    config = dict(template.probe_config)
    config.update(
        {
            "map_dir": template.template_map_path,
            "num_maps": 1,
            "num_agents": template.template_agent_count,
            "min_agents_per_env": template.template_agent_count,
            "max_agents_per_env": int(template.probe_config["max_agents_per_env"]),
            "starting_map": 0,
            "seed": scene_seed,
            "init_step": int(template.probe_env_kwargs["init_step"]),
            "init_step_spread": False,
            "capture_final_observations": True,
            "eval_mode": 0,
        }
    )
    env = Drive(**config)
    drive_module.binding.vec_reset(env.c_envs, [scene_seed])
    env.tick = 0
    env.truncations.fill(0)
    if env.capture_final_observations:
        env.final_observation_valid.fill(0)
    return env


@pytest.mark.skipif(not MAP_PATH.is_file(), reason="CARLA test map is not available")
def test_private_scene_copies_replay_prefix_and_hold_timeout_matches_production():
    from pufferlib.ocean.drive import drive as drive_module

    ensure_private_binding()
    template = create_template(_env_config(), seed=101)
    production = None
    batch = None
    scene_seed = 7301
    try:
        production = _production_env(template, scene_seed)
        batch = SceneBatch.from_template(
            template,
            template.template_map_path,
            template.template_agent_count,
            copies=2,
            scene_seed=scene_seed,
        )
        assert batch.observations.shape == (2, template.template_agent_count, template.num_obs)
        assert np.array_equal(batch.observations[0], batch.observations[1])
        initial = batch.metadata()
        assert np.array_equal(batch.observations[0], production.observations)
        assert initial["eligible"].shape == (2, template.template_agent_count)
        assert initial["reward_coefficients"].shape == (2, template.template_agent_count, 17)
        assert initial["traffic_control_pose"].shape[0] == 2
        assert initial["scene_active_agent_count"].tolist() == [template.template_agent_count] * 2
        assert batch.fingerprint(0) == batch.fingerprint(1)

        prefix = np.array([[7] * template.template_agent_count], dtype=np.int32)
        fingerprints = batch.replay_prefix(prefix, frame_idx=1)
        production.step(np.repeat(batch.action_table[7][None, :], template.template_agent_count, axis=0))
        assert len(set(fingerprints)) == 1
        before_boundary = batch.metadata()
        assert before_boundary["timestep"].tolist() == [1, 1]

        action = batch.action_table[7][None, None, :].repeat(
            2, axis=0
        ).repeat(template.template_agent_count, axis=1)
        private_step = batch.step(action)
        production_step = production.step(np.repeat(batch.action_table[7][None, :], template.template_agent_count, axis=0))
        assert np.array_equal(private_step["observations"][0], production_step[0])
        assert np.array_equal(private_step["rewards"][0], production_step[1])
        assert np.array_equal(private_step["terminals"][0], production_step[2])
        assert np.array_equal(private_step["truncations"][0], production_step[3])

        final_private = batch.step(action)
        final_production = production.step(
            np.repeat(batch.action_table[7][None, :], template.template_agent_count, axis=0)
        )
        assert final_private["metadata"]["scene_done"].tolist() == [True, True]
        assert np.array_equal(final_private["truncations"][0], final_production[3])
        valid = production.final_observation_valid.astype(bool)
        expected_final_observation = production.final_observations.copy()
        assert np.all(valid)
        assert np.array_equal(final_private["observations"][0], expected_final_observation)
        held_metadata = batch.metadata()
        held_digest = batch.fingerprint(0)
        held = batch.step(action)
        after_hold = batch.metadata()
        assert np.array_equal(after_hold["pose_xyz"], held_metadata["pose_xyz"])
        assert np.array_equal(after_hold["timestep"], held_metadata["timestep"])
        assert np.array_equal(after_hold["episode_seed"], held_metadata["episode_seed"])
        assert np.all(held["rewards"] == 0)
        assert np.all(~held["truncations"])
        assert batch.fingerprint(0) != held_digest  # transition buffers are cleared on the next held call

        batch.reset()
        reset_metadata = batch.metadata()
        assert reset_metadata["scene_done"].tolist() == [False, False]
        assert reset_metadata["timestep"].tolist() == [0, 0]
        assert batch.fingerprint(0) == batch.fingerprint(1)
        batch.step(action)
        assert batch.metadata()["timestep"].tolist() == [1, 1]

        native_digest = batch.fingerprint(0)
        generators = batch.set_continuation_seeds([3, 4])
        assert generators[0].integers(0, 2**31) != generators[1].integers(0, 2**31)
        assert batch.fingerprint(0) == native_digest  # actor RNG setup does not touch native state
    finally:
        if batch is not None:
            batch.close()
        if production is not None:
            production.close()
        template.close()


def test_private_build_has_separate_module_and_expected_normalized_jerk_table():
    from project.soft_actor_critic_q_calibration.probe_env import _jerk_action_table

    module = ensure_private_binding()
    assert module.__name__ == "probe_binding"
    table = _jerk_action_table()
    assert table.shape == (12, 2)
    assert table[:, 0].tolist()[:3] == [-1.0, -1.0, -1.0]
    assert table[:, 1].tolist()[::3] == [-1.0, -1.0, -1.0, -1.0]
