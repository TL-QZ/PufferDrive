"""Static workflow checks plus an opt-in, single-update CPU checkpoint integration."""

import copy
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_config import PROJECT, cli_values, finetune_overrides, validate_resume
from profile_nuplan_dt03 import qualifying, select_candidate
from pufferlib import pufferl as training
from pufferlib.ocean.evaluation_utils import evaluation_utils as benchmark


def resolved_config(overrides=None):
    config = finetune_overrides()
    config.update(overrides or {})
    with patch("sys.argv", ["puffer", *cli_values(config)]):
        return dict(training.load_config("puffer_drive"))


def test_ddp_config_counts():
    with patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "2"}):
        config = resolved_config()
    env, vec, train = config["env"], config["vec"], config["train"]
    assert env["dt"] == 0.3 and env["scenario_length"] == 66 and env["resample_replay_to_dt"]
    assert vec["num_workers"] == 20
    assert env["num_agents"] * vec["num_envs"] == 640
    assert env["num_agents"] * vec["batch_size"] == 320
    assert 640 * train["bptt_horizon"] == 640000
    assert train["minibatch_size"] // train["max_minibatch_size"] == 4
    assert train["max_minibatch_size"] % train["bptt_horizon"] == 0
    assert env["num_maps"] == 169715
    assert train["total_timesteps"] == 500000000
    assert config["rnn_name"] is None and not train["use_rnn"]


@pytest.mark.parametrize("route", ["periodic", "standalone", "failure_replay"])
def test_evaluation_overrides_checkpoint_timing(tmp_path, route):
    base = resolved_config({"eval.num_agents": 300})
    (tmp_path / "final_model.pt").touch()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(base))
    base["load_model_path"] = str(tmp_path / "final_model.pt")
    # Periodic eval receives the live policy; standalone/render reload its config.
    if route != "periodic":
        base, _ = benchmark.load_checkpoint_architecture(base)
    environment, benchmarks = benchmark.load_benchmark_config(
        str(PROJECT / "override_config/evaluation_benchmarks.yaml"),
        ["carla", "carla_fast", "nuplan_single", "nuplan_multi"],
    )
    for item in benchmarks:
        args = benchmark.build_benchmark_args(base, item, environment)
        assert args["env"]["dt"] == 0.1
        assert args["env"]["resample_replay_to_dt"] is False
        if route == "failure_replay":
            workers, _ = benchmark._plan_failure_replay_workers(args, [(0, 42)], 1, args["env"]["scenario_length"])
        else:
            workers, _ = benchmark._plan_benchmark_eval_workers(args, 1, 1, args["env"]["scenario_length"])
        assert workers[0]["dt"] == 0.1 and workers[0]["resample_replay_to_dt"] is False


def test_resume_rejects_old_timing_and_resource_changes(tmp_path):
    with patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "2"}):
        saved = resolved_config({"train.data_dir": str(tmp_path)})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(saved))
    validate_resume(tmp_path, finetune_overrides())
    for section, field, value in [("env", "dt", 0.1), ("env", "resample_replay_to_dt", False),
                                  ("env", "num_agents", 16), ("train", "bptt_horizon", 200)]:
        incompatible = copy.deepcopy(saved)
        incompatible[section][field] = value
        path.write_text(yaml.safe_dump(incompatible))
        with pytest.raises(ValueError, match="Incompatible resume"):
            validate_resume(tmp_path, finetune_overrides())


def test_profile_selection_uses_fastest_and_smaller_ties():
    results = [{"agents": agents, "qualifies": True, "global_transitions_per_second": rate}
               for agents, rate in [(16, 100), (32, 104), (64, 108)]]
    assert select_candidate(results)["agents"] == 32
    results[2]["qualifies"] = False
    assert select_candidate(results)["agents"] == 16
    assert select_candidate([]) is None


def test_profile_rejects_incomplete_oom_swap_and_unsynchronized_updates():
    row = dict(finite_losses=True, synchronized_updates=True, synchronized_retained_counts=True,
               optimizer_updates=2, retained_transitions=100, truncation_count=2,
               all_truncations_terminal=True, cuda_peak_reserved_bytes=80,
               device_total_bytes=100, device_free_bytes=20)
    rows = [[copy.deepcopy(row) for _ in range(3)] for _ in range(2)]
    host = dict(min_available_fraction=0.3, swap_growth_bytes=0)
    assert qualifying(rows, host)
    assert not qualifying(rows[:1], host)
    assert not qualifying(rows, dict(host, swap_growth_bytes=1))
    assert not qualifying(rows, dict(host, min_available_fraction=0.19))
    rows[1][2]["synchronized_updates"] = False
    assert not qualifying(rows, host)


def test_train_eval_render_launcher_dry_runs(tmp_path):
    project = tmp_path / "project" / PROJECT.name
    shutil.copytree(PROJECT, project, ignore=shutil.ignore_patterns("__pycache__", "profiling"))
    (tmp_path / ".venv").symlink_to(PROJECT.parents[1] / ".venv", target_is_directory=True)
    experiment = tmp_path / "experiments" / PROJECT.name
    source = experiment / f"{PROJECT.name}_source_seed0"
    run_name = f"{PROJECT.name}_nuplan_sdc_finetune_dt03_test_seed0"
    run = experiment / "nuplan_sdc_finetune_dt03" / run_name
    for directory in (source, run):
        directory.mkdir(parents=True)
        (directory / "final_model.pt").touch()
        (directory / "config.yaml").write_text("{}\n")
    metrics = run / "eval/nuplan_single_final_model_mean_metrics/test/episode_metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("scenario,episode_seed\n0,42\n")
    commands = [("train/launch_nuplan_sdc_finetune_2gpu.sh", ["0"], "0,1"),
                ("eval/evaluate_nuplan_sdc_finetune.sh", ["0"], "0"),
                ("render/render_nuplan_sdc_finetune_failures.sh", ["0", "nuplan_single"], "0")]
    for script, arguments, devices in commands:
        result = subprocess.run(["bash", str(project / script), *arguments], text=True, capture_output=True,
                                env=dict(os.environ, DRY_RUN="1", CUDA_VISIBLE_DEVICES=devices), check=True)
        assert "nuplan_sdc_finetune_dt03" in result.stdout
        assert "Command:" in result.stdout
    assert not (run / "initial_checkpoint").exists()
    assert not (metrics.parent / "failure_analysis").exists()


@pytest.mark.skipif(not os.getenv("DT03_TEST_SOURCE_RUN"), reason="requires the real CARLA checkpoint")
def test_cpu_checkpoint_rollout_update_and_resume(tmp_path, monkeypatch):
    source = Path(os.environ["DT03_TEST_SOURCE_RUN"]).resolve()
    staged = tmp_path / "initial_checkpoint"
    staged.mkdir()
    for name in ("final_model.pt", "config.yaml"):
        (staged / name).symlink_to(source / name)
    config = resolved_config({
        "load_model_path": str(staged / "final_model.pt"), "train.data_dir": str(tmp_path / "run"),
        "env.map_dir": "pufferlib/resources/drive/binaries/nuplan", "env.num_maps": 1, "env.num_agents": 1,
        "vec.backend": "Serial", "vec.num_envs": 2, "vec.num_workers": 2, "vec.batch_size": 2,
        "train.device": "cpu", "train.compile": False, "train.precision": "float32", "train.amp": False,
        "train.bptt_horizon": 132, "train.minibatch_size": 132, "train.max_minibatch_size": 132,
        "train.update_epochs": 1, "train.evaluation_interval_epochs": None,
        "wandb": False, "neptune": False, "tb": False,
    })
    torch.set_num_threads(1)
    evidence = {}

    class Finished(Exception):
        pass

    class CheckedPPO(training.PuffeRL):
        def print_dashboard(self, *args, **kwargs):
            pass

        def train(self):
            assert self.global_step == 264 and self.epoch == 0 and not self.optimizer.state
            assert self.truncations.count_nonzero() >= 2
            assert torch.all(self.terminals[self.truncations.bool()] == 1)
            initial = next(self.uncompiled_policy.parameters()).detach().clone()
            super().train()
            assert self.optimizer.state
            assert not torch.equal(initial, next(self.uncompiled_policy.parameters()))
            assert all(math.isfinite(self.losses[key]) for key in ("policy_loss", "value_loss", "entropy"))
            self.save_checkpoint()
            state = torch.load(Path(self.config["data_dir"]) / "trainer_state.pt", map_location="cpu", weights_only=False)
            self.load_training_state(Path(self.config["data_dir"]) / "trainer_state.pt")
            assert self.epoch == state["epoch"] == 1 and self.global_step == 264
            evidence.update(parameters=self.model_size, transitions=self.global_step, epoch=self.epoch)
            raise Finished()

    monkeypatch.setattr(training, "PuffeRL", CheckedPPO)
    with pytest.raises(Finished):
        training.train("puffer_drive", args=config)
    assert evidence["transitions"] == 264
