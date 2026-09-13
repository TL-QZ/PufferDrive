"""CPU contract checks for scratch initialization, resume, controllers and timing."""

import copy
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import psutil
import yaml

from selfplay_config import resolve_config, selfplay_overrides, validate_resume
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.evaluation_utils import evaluation_utils as benchmarks


def test_scratch_and_benchmark_timing():
    args = resolve_config(selfplay_overrides())
    assert args["load_model_path"] is None and args["load_id"] is None
    assert args["rnn_name"] is None
    assert args["env"]["action_type"] == args["policy"]["action_type"] == "discrete"
    assert args["train"]["total_timesteps"] == 10_000_000_000
    common, selected = benchmarks.load_benchmark_config(
        args["eval"]["benchmark_config"], args["train"]["evaluation_benchmarks"])
    assert [benchmark["name"] for benchmark in selected] == ["carla_fast", "nuplan_single", "nuplan_multi"]
    for benchmark in selected:
        resolved = benchmarks.build_benchmark_args(args, benchmark, common)
        assert resolved["env"]["dt"] == 0.1
        assert resolved["env"]["resample_replay_to_dt"] is False
        assert resolved["env"]["num_agents"] == 300
        assert resolved["env"]["resample_frequency"] == resolved["env"]["scenario_length"]


@pytest.mark.parametrize("changed", [None, "timing", "architecture", "resources", "seed", "budget", "directory"])
def test_resume_checks_resolved_config(tmp_path, changed):
    args = resolve_config(selfplay_overrides())
    args["train"]["data_dir"] = str(tmp_path)
    args["run_name"] = tmp_path.name
    saved = copy.deepcopy(dict(args))
    saved["train"]["total_timesteps"] //= 2
    if changed == "timing":
        saved["env"]["dt"] = 0.1
    elif changed == "architecture":
        saved["policy"]["backbone_hidden_size"] = 512
    elif changed == "resources":
        saved["vec"]["num_envs"] += 2
    elif changed == "seed":
        saved["train"]["seed"] += 1
    elif changed == "budget":
        saved["train"]["total_timesteps"] //= 2
    elif changed == "directory":
        saved["train"]["data_dir"] = str(tmp_path / "other")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(saved))
    if changed is None:
        validate_resume(tmp_path, args)
    else:
        with pytest.raises(ValueError, match="Incompatible resume"):
            validate_resume(tmp_path, args)


@pytest.mark.skipif(not Path("data/nuplan_train").is_dir(), reason="requires local nuPlan training binaries")
def test_selfplay_controllers_and_resampling():
    args = resolve_config(selfplay_overrides())
    env = Drive(**dict(args["env"], num_agents=128, seed=42))
    try:
        observations, _ = env.reset(seed=42)
        initial_maps = tuple(env.map_ids)
        scenes = env.get_state()
        assert any(scene["active_agent_count"] > 1 for scene in scenes)
        assert sum(scene["active_agent_count"] for scene in scenes) == 128
        for scene in scenes:
            for agent_idx in scene["active_agent_indices"]:
                agent = scene["agents"][agent_idx]
                assert agent["controller"] == binding.CONTROLLER_POLICY
                assert len(agent["log_trajectory_x"]) == 67
        for step in range(1, 133):
            observations, rewards, _, truncations, _ = env.step(np.full(env.actions.shape, 4, dtype=env.actions.dtype))
            assert np.isfinite(observations).all() and np.isfinite(rewards).all()
            if step in (66, 132):
                assert truncations.all()
        assert tuple(env.map_ids) != initial_maps
    finally:
        env.close()


def test_probe_cleanup_after_leader_exits():
    from profile_nuplan_selfplay import stop_candidate

    command = "import subprocess; child = subprocess.Popen(['sleep', '120']); print(child.pid, flush=True)"
    leader = subprocess.Popen([sys.executable, "-c", command], start_new_session=True,
                              stdout=subprocess.PIPE, text=True)
    try:
        child_pid = int(leader.stdout.readline())
        leader.wait(timeout=5)
        stop_candidate(leader)
        for _ in range(20):
            try:
                status = psutil.Process(child_pid).status()
            except psutil.NoSuchProcess:
                break
            if status == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.1)
        else:
            pytest.fail("Probe child survived cleanup after its leader exited")
    finally:
        stop_candidate(leader)
        leader.stdout.close()
