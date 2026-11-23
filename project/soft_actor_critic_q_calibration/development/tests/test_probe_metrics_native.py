"""CPU checks for read-only native interval metrics exports."""

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
METRIC_KEYS = {
    "n",
    "offroad_rate",
    "episode_length",
    "collision_rate",
    "episode_return",
    "red_light_violation_rate",
    "comfort_violation_count",
    "velocity_progress_sum",
    "num_goals_reached",
    "lane_center_rate",
    "dnf_rate",
    "score",
    "avg_speed_per_agent",
    "avg_distance_per_infraction",
    "reward_components/collision",
    "reward_components/offroad",
    "reward_components/red_light",
    "reward_components/goal",
    "reward_components/lane_align",
    "reward_components/lane_center",
    "reward_components/comfort",
    "reward_components/velocity",
    "reward_components/timestep",
    "reward_components/reverse",
    "reward_components/overspeed",
    "reward_components/ade",
    "at_fault_collision_rate",
    "puffer_score",
    "driving_direction_score",
    "speed_limit_compliance",
    "making_progress_rate",
    "progress_ratio",
    "comfort_score",
    "multi_lane_time",
    "multi_lane_score",
    "active_agent_count",
    "episode_timestep",
    "agents_per_batch",
    "total_distance_travelled",
    "total_infractions",
}


def _env_config(compute_eval_metrics=True):
    return {
        "render_mode": "headless",
        "report_interval": 1,
        "dt": 0.3,
        "scenario_length": 50,
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
        "compute_eval_metrics": compute_eval_metrics,
        "eval_mode": 0,
        "min_agents_per_env": 4,
        "max_agents_per_env": 4,
    }


def _batch(template, copies=2, scene_seed=7301):
    return SceneBatch.from_template(
        template,
        template.template_map_path,
        template.template_agent_count,
        copies=copies,
        scene_seed=scene_seed,
    )


@pytest.mark.skipif(not MAP_PATH.is_file(), reason="CARLA test map is not available")
def test_native_interval_metrics_match_observed_interval_and_keep_exact_schema():
    ensure_private_binding()
    template = create_template(_env_config(), seed=101)
    batch = None
    try:
        batch = _batch(template, copies=2)
        action = np.repeat(batch.action_table[10][None, None, :], batch.copies * batch.agent_count, axis=0)
        action = action.reshape(batch.copies, batch.agent_count, 2)

        # Give the baseline a nonzero distance prefix, then measure only the later rollout.
        prefix = batch.step(action)
        prefix_metadata = prefix["metadata"]
        candidates = np.flatnonzero(
            prefix_metadata["eligible"][0] & (prefix_metadata["speed_mps"][0] > 0.1)
        )
        assert candidates.size > 0, "test scene must have a moving eligible ego after its prefix"
        ego_idx = int(candidates[0])
        assert prefix_metadata["timestep"].tolist() == [1, 1]

        baseline = batch.binding.vec_begin_probe_metrics(batch.c_envs, ego_idx)
        assert type(baseline).__name__ == "PyCapsule"
        empty_interval_rows = batch.binding.vec_get_probe_metrics(
            batch.c_envs, baseline, np.zeros((batch.copies, 4), dtype=np.bool_)
        )
        assert all(row["progress_ratio"] == 0.0 for row in empty_interval_rows)
        assert all(row["making_progress_rate"] == 0.0 for row in empty_interval_rows)

        interval_steps = 20
        expected_return = np.zeros(batch.copies, dtype=np.float64)
        expected_goal_count = np.zeros(batch.copies, dtype=np.int64)
        expected_distance = np.zeros(batch.copies, dtype=np.float64)
        for _ in range(interval_steps):
            metadata_before = batch.metadata()
            step_result = batch.step(action)
            expected_return += step_result["rewards"][:, ego_idx]
            expected_goal_count += step_result["events"]["reached_goal"][:, ego_idx]
            moved_before = metadata_before["eligible"][:, ego_idx]
            expected_distance += (
                moved_before * step_result["metadata"]["speed_mps"][:, ego_idx] * template.dt
            )

        before_read = [batch.fingerprint(copy_idx) for copy_idx in range(batch.copies)]
        metadata_before_read = batch.metadata()
        # These are the caller's interval ORs. Deliberately distinct values ensure
        # the native export does not reuse sticky Log rates from the prefix.
        interval_events = np.array(
            [[False, False, False, False], [True, True, True, True]], dtype=np.bool_
        )
        rows = batch.binding.vec_get_probe_metrics(batch.c_envs, baseline, interval_events)
        after_read = [batch.fingerprint(copy_idx) for copy_idx in range(batch.copies)]
        metadata_after_read = batch.metadata()

        assert len(rows) == batch.copies
        for row in rows:
            assert set(row) == METRIC_KEYS
            assert all(np.isfinite(value) for value in row.values())
            assert row["n"] == 1.0
            assert row["episode_length"] == interval_steps
            assert row["episode_timestep"] == interval_steps
            assert row["active_agent_count"] == batch.agent_count
            assert row["agents_per_batch"] == batch.copies * batch.agent_count
        np.testing.assert_allclose([row["episode_return"] for row in rows], expected_return, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal([row["num_goals_reached"] for row in rows], expected_goal_count)
        np.testing.assert_allclose(
            [row["total_distance_travelled"] for row in rows], expected_distance, rtol=1e-5, atol=1e-3
        )
        assert prefix_metadata["speed_mps"][0, ego_idx] * template.dt > 0.0
        assert rows[0]["collision_rate"] == 0.0
        assert rows[0]["at_fault_collision_rate"] == 0.0
        assert rows[0]["offroad_rate"] == 0.0
        assert rows[0]["red_light_violation_rate"] == 0.0
        assert rows[0]["total_infractions"] == 0.0
        assert rows[1]["collision_rate"] == 1.0
        assert rows[1]["at_fault_collision_rate"] == 1.0
        assert rows[1]["offroad_rate"] == 1.0
        assert rows[1]["red_light_violation_rate"] == 1.0
        assert rows[1]["total_infractions"] == 1.0
        assert rows[1]["puffer_score"] == 0.0
        assert rows[0]["progress_ratio"] > 0.2
        assert rows[0]["making_progress_rate"] == 1.0
        assert before_read == after_read
        for key in metadata_before_read:
            np.testing.assert_array_equal(metadata_before_read[key], metadata_after_read[key])
    finally:
        if batch is not None:
            batch.close()
        template.close()


@pytest.mark.skipif(not MAP_PATH.is_file(), reason="CARLA test map is not available")
def test_native_interval_metrics_reject_bad_sources_ego_slots_and_event_arrays():
    binding = ensure_private_binding()
    template = create_template(_env_config(), seed=202)
    batch = None
    other_batch = None
    wrong_count_batch = None
    disabled_template = None
    disabled_batch = None
    try:
        batch = _batch(template, copies=2, scene_seed=7302)
        with pytest.raises(ValueError, match="ego_idx"):
            binding.vec_begin_probe_metrics(batch.c_envs, -1)
        with pytest.raises(ValueError, match="ego slot"):
            binding.vec_begin_probe_metrics(batch.c_envs, batch.agent_count)
        with pytest.raises(TypeError, match="ego_idx"):
            binding.vec_begin_probe_metrics(batch.c_envs, True)

        capsule = binding.vec_begin_probe_metrics(batch.c_envs, 0)
        other_batch = _batch(template, copies=2, scene_seed=7303)
        event_flags = np.zeros((2, 4), dtype=np.bool_)
        with pytest.raises(ValueError, match="different vector environment"):
            binding.vec_get_probe_metrics(other_batch.c_envs, capsule, event_flags)
        wrong_count_batch = _batch(template, copies=1, scene_seed=7305)
        with pytest.raises(ValueError, match="different vector environment"):
            binding.vec_get_probe_metrics(
                wrong_count_batch.c_envs, capsule, np.zeros((1, 4), dtype=np.bool_)
            )
        with pytest.raises(TypeError, match="capsule"):
            binding.vec_get_probe_metrics(batch.c_envs, object(), event_flags)
        with pytest.raises(ValueError, match="contiguous native bool"):
            binding.vec_get_probe_metrics(batch.c_envs, capsule, np.zeros((2, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="contiguous native bool"):
            binding.vec_get_probe_metrics(batch.c_envs, capsule, np.zeros((2, 8), dtype=np.bool_)[:, ::2])
        with pytest.raises(ValueError, match="shape"):
            binding.vec_get_probe_metrics(batch.c_envs, capsule, np.zeros((2, 3), dtype=np.bool_))

        disabled_template = create_template(_env_config(compute_eval_metrics=False), seed=303)
        disabled_batch = _batch(disabled_template, copies=1, scene_seed=7304)
        with pytest.raises(ValueError, match="compute_eval_metrics"):
            binding.vec_begin_probe_metrics(disabled_batch.c_envs, 0)
    finally:
        if disabled_batch is not None:
            disabled_batch.close()
        if disabled_template is not None:
            disabled_template.close()
        if wrong_count_batch is not None:
            wrong_count_batch.close()
        if other_batch is not None:
            other_batch.close()
        if batch is not None:
            batch.close()
        template.close()
