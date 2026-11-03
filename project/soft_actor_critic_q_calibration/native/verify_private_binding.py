"""CPU-only smoke and transition-parity checks for the private diagnostic binding."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from project.soft_actor_critic_q_calibration.diagnostic import (
    get_diagnostics,
    install_private_binding,
)
import pufferlib.ocean.drive.drive as drive_module


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAP_FILE = sorted((PROJECT_ROOT / "data" / "nuplan_val").glob("*.bin"))[0]
PRODUCTION_BINDING = drive_module.binding
PRIVATE_BINDING = install_private_binding()


def make_env(**overrides):
    arguments = {
        "render_mode": "headless",
        "num_agents": 1,
        "min_agents_per_env": 1,
        "max_agents_per_env": 1,
        "num_maps": 1,
        "map_dir": str(MAP_FILE),
        "simulation_mode": "gigaflow",
        "goal_source": "map",
        "goal_regen_mode": "finite",
        "num_goals": 3,
        "min_goal_spacing": 20.0,
        "max_goal_spacing": 60.0,
        "scenario_length": 2560,
        "dt": 0.3,
        "seed": 23,
        "compute_eval_metrics": False,
    }
    arguments.update(overrides)
    return drive_module.Drive(**arguments)


def check_padded_rows():
    env = make_env(
        goal_source="route",
        min_goal_spacing=100000.0,
        max_goal_spacing=100001.0,
    )
    try:
        observations, _ = env.reset(seed=91)
        metadata = get_diagnostics(env)
        assert observations.shape[0] == env.num_agents == 1
        assert metadata["slot_idx"].tolist() == [0]
        assert metadata["agent_id"].tolist() == [-1]
        assert metadata["episode_active"].tolist() == [0]
        assert metadata["graph_distance_status"].tolist() == [6]
        assert metadata["reward_coefficients"].shape == (1, 17)
    finally:
        env.close()


def check_timeout_reset():
    env = make_env(
        scenario_length=2,
        capture_final_observations=True,
    )
    try:
        env.reset(seed=91)
        before = get_diagnostics(env)
        action = np.zeros((env.num_agents,), dtype=np.int64)
        env.step(action)
        _, _, terminals, truncations, _ = env.step(action)
        after = get_diagnostics(env)
        assert terminals.tolist() == [False]
        assert truncations.tolist() == [True]
        assert env.final_observation_valid.tolist() == [1]
        assert after["episode_seed"][0] != before["episode_seed"][0]
        assert after["timestep"].tolist() == [0]
    finally:
        env.close()


def check_multi_scene_rows():
    env = make_env(
        num_agents=2,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=2,
        map_dir=str(PROJECT_ROOT / "data" / "nuplan_val"),
    )
    try:
        env.reset(seed=91)
        metadata = get_diagnostics(env)
        assert env.num_envs == 2
        assert metadata["scene_idx"].tolist() == [0, 1]
        assert metadata["slot_idx"].tolist() == [0, 0]
        assert metadata["episode_seed"].shape == (env.num_agents,)
        padded_rows = metadata["agent_id"] == -1
        assert np.all(metadata["episode_active"][padded_rows] == 0)
        assert np.all(metadata["graph_distance_status"][padded_rows] == 6)
    finally:
        env.close()


def run_trajectory(binding):
    drive_module.binding = binding
    env = make_env(reward_conditioning=True, reward_randomization=True)
    try:
        observation, _ = env.reset(seed=91)
        trajectory = [("observation", observation.copy())]
        metadata_rows = []
        for action in (0, 31, 62):
            if binding is PRIVATE_BINDING:
                metadata = get_diagnostics(env)
                assert metadata["slot_idx"].tolist() == [0]
                assert metadata["timestep"].shape == (env.num_agents,)
                metadata_rows.append(metadata)
            observation, rewards, terminals, truncations, _ = env.step(
                np.array([action], dtype=np.int64)
            )
            trajectory.extend(
                (
                    ("observation", observation.copy()),
                    ("rewards", rewards.copy()),
                    ("terminals", terminals.copy()),
                    ("truncations", truncations.copy()),
                )
            )
        return trajectory, metadata_rows
    finally:
        env.close()


def check_transition_parity():
    production, _ = run_trajectory(PRODUCTION_BINDING)
    private, metadata_rows = run_trajectory(PRIVATE_BINDING)
    assert len(production) == len(private)
    for index, ((production_name, production_array), (private_name, private_array)) in enumerate(
        zip(production, private)
    ):
        assert production_name == private_name
        assert np.array_equal(production_array, private_array), (
            f"transition parity failed at output {index} ({production_name})"
        )
    assert len(metadata_rows) == 3


def main():
    print(f"map_file={MAP_FILE.relative_to(PROJECT_ROOT)}")
    check_padded_rows()
    print("padded_row_smoke=PASS (one observation row, inactive sentinel metadata)")
    check_timeout_reset()
    print("timeout_reset_smoke=PASS (scenario_length=2 test-only override)")
    check_multi_scene_rows()
    print("multi_scene_row_order=PASS (two observation rows across two scenes)")
    check_transition_parity()
    print("production_private_parity=PASS (initial observation + 3 action transitions, bitwise exact)")


if __name__ == "__main__":
    main()
