"""CPU replay checks. Capture the default path BEFORE rebuilding changed C code.

REPLAY_BASELINE=/tmp/replay_before.npz python -m pytest ... compares that capture.
The capture is deliberately external: tests never regenerate an expected result.
"""

import argparse
import os
import struct
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import binding
from pufferlib.pufferl import load_config

REPO = Path(__file__).resolve().parents[2]
NUPLAN = REPO / "pufferlib/resources/drive/binaries/nuplan"
CARLA = REPO / "pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin"


def replay_config(**overrides):
    with patch("sys.argv", ["puffer"]):
        config = load_config("puffer_drive")["env"]
    config.update(
        map_dir=str(NUPLAN), num_maps=1, num_agents=1,
        simulation_mode="replay", control_mode="control_sdc_only",
        sdc_controller="policy", non_sdc_controller="replay", non_vehicle_controller="replay",
        dt=0.1, scenario_length=66, resample_frequency=66,
        init_step=0, init_step_spread=False, seed=42,
    )
    config.update(overrides)
    return config


def default_capture(option=None):
    captured = {}
    for mode in ("gigaflow", "replay"):
        config = replay_config()
        config.pop("resample_replay_to_dt", None)
        if option is not None:
            config["resample_replay_to_dt"] = option
        if mode == "gigaflow":
            config.update(map_dir=str(CARLA), simulation_mode=mode, num_agents=16,
                          min_agents_per_env=16, max_agents_per_env=16,
                          control_mode="control_vehicles", non_sdc_controller="policy", dt=0.3)
        env = Drive(**config)
        try:
            observations, _ = env.reset(seed=42)
            rows = [np.concatenate((observations.ravel(), env.rewards, env.terminals, env.truncations))]
            for step_idx in range(140):
                actions = np.full(env.actions.shape, 31 + step_idx % 3, dtype=env.actions.dtype)
                observations, rewards, terminals, truncations, _ = env.step(actions)
                rows.append(np.concatenate((observations.ravel(), rewards, terminals, truncations)))
            captured[mode] = np.stack(rows)
        finally:
            env.close()
    return captured


def test_default_path_omitted_equals_false():
    omitted = default_capture()
    explicit = default_capture(False)
    for mode in omitted:
        np.testing.assert_array_equal(omitted[mode], explicit[mode])


@pytest.mark.skipif(not os.getenv("REPLAY_BASELINE"), reason="requires a pre-change capture")
def test_default_path_matches_before_change():
    expected = np.load(os.environ["REPLAY_BASELINE"])
    for mode, values in default_capture(False).items():
        np.testing.assert_array_equal(values, expected[mode])


@pytest.mark.parametrize("overrides,field", [
    ({"dt": 0.25}, "dt/log_dt"), ({"dt": 0.05}, "dt/log_dt"),
    ({"dt": float("nan")}, "dt"), ({"dt": float("inf")}, "dt"),
    ({"dt": 0}, "dt"), ({"dt": -0.3}, "dt"),
    ({"simulation_mode": "gigaflow"}, "simulation_mode"),
    ({"init_step": 1}, "init_step"), ({"init_step_spread": True}, "init_step_spread"),
    ({"scenario_length": 67}, "insufficient"),
])
def test_invalid_creation(overrides, field):
    config = replay_config(resample_replay_to_dt=True, dt=0.3)
    config.update(overrides)
    with pytest.raises(ValueError, match=field):
        Drive(**config)


@pytest.mark.parametrize("damage,field", [
    ("log_dt", "log_dt"), ("log_length", "trajectory_size"),
    ("negative_count", "count"), ("negative_trajectory", "count"),
    ("truncated", "truncated"), ("trailing", "trailing"),
])
def test_rejected_binary_releases_partial_load(tmp_path, damage, field):
    source = next(NUPLAN.glob("*.bin")).read_bytes()
    data = bytearray(source)
    # This fixture ends with log_length, log_dt, and two empty metadata lists.
    assert struct.unpack_from("<ifii", data, len(data) - 16) == (201, np.float32(0.1), 0, 0)
    if damage == "log_dt":
        struct.pack_into("<f", data, len(data) - 12, float("nan"))
    elif damage == "log_length":
        struct.pack_into("<i", data, len(data) - 16, 200)
    elif damage == "negative_count":
        struct.pack_into("<i", data, 0, -1)
    elif damage == "negative_trajectory":
        struct.pack_into("<i", data, 24, -1)
    elif damage == "truncated":
        data = data[:-1]
    elif damage == "trailing":
        data.extend(b"extra")
    path = tmp_path / "invalid.bin"
    path.write_bytes(data)
    live_before = binding.map_cache_live_count()
    for _ in range(3):
        with pytest.raises(ValueError, match=field) as error:
            Drive(**replay_config(map_dir=str(path), resample_replay_to_dt=True, dt=0.3))
        assert str(path) in str(error.value)
    assert binding.map_cache_live_count() == live_before


def test_export_and_map_switches(tmp_path):
    source = next(NUPLAN.glob("*.bin")).read_bytes()
    for map_idx in range(2):
        (tmp_path / f"scenario{map_idx}.bin").write_bytes(source)
    env = Drive(**replay_config(map_dir=str(tmp_path), num_maps=2, num_agents=2,
                               resample_replay_to_dt=True, dt=0.3))
    try:
        env.reset(seed=42)
        for step_idx in range(133):
            if step_idx % 66 == 0:
                trajectories = env.get_ground_truth_trajectories()
                assert trajectories["x"].shape == (2, 1, 67)
                assert trajectories["valid"].shape == (2, 1, 67)
                assert np.isfinite(trajectories["x"]).all()
            _, _, _, truncations, _ = env.step(np.full(env.actions.shape, 31, dtype=env.actions.dtype))
            if (step_idx + 1) % 66 == 0:
                assert (truncations == 1).all()
        outputs = [np.zeros((2, 66), np.float32) for _ in range(4)]
        outputs += [np.zeros((2, 66), np.int32), np.zeros(2, np.int32), np.zeros(2, np.int32)]
        with pytest.raises(ValueError, match="67 states"):
            binding.vec_get_global_ground_truth_trajectories(env.c_envs, *outputs)
    finally:
        env.close()


def test_actual_load_failure_releases_earlier_instances(monkeypatch):
    original_init = binding.env_init
    created = 0
    live_before = binding.map_cache_live_count()

    def reject_second(*args, **kwargs):
        nonlocal created
        created += 1
        if created == 2:
            kwargs["dt"] = 0.25
        return original_init(*args, **kwargs)

    monkeypatch.setattr(binding, "env_init", reject_second)
    with pytest.raises(ValueError, match="dt/log_dt"):
        Drive(**replay_config(num_agents=2, resample_replay_to_dt=True, dt=0.3))
    assert created == 2
    assert binding.map_cache_live_count() == live_before


def test_map_switch_failure_is_closed_and_cannot_step(tmp_path):
    path = tmp_path / "scenario.bin"
    path.write_bytes(next(NUPLAN.glob("*.bin")).read_bytes())
    env = Drive(**replay_config(map_dir=str(path), resample_replay_to_dt=True, dt=0.3))
    try:
        env.reset(seed=42)
        actions = np.full(env.actions.shape, 31, dtype=env.actions.dtype)
        for _ in range(65):
            env.step(actions)
        path.write_bytes(b"truncated")
        with pytest.raises(ValueError, match="Scenario"):
            env.step(actions)
        with pytest.raises(RuntimeError, match="rejected"):
            env.step(actions)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-baseline", required=True)
    output = Path(parser.parse_args().capture_baseline)
    if output.exists():
        raise SystemExit(f"Refusing to replace baseline: {output}")
    np.savez_compressed(output, **default_capture())
